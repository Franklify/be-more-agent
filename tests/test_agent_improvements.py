import importlib
import json
import sys
import threading
import unittest
from unittest import mock

try:
    import ollama as real_ollama
except ModuleNotFoundError:
    real_ollama = None


class SearchError(Exception):
    pass


fake_modules = {
    name: mock.MagicMock()
    for name in (
        "tkinter", "tkinter.ttk", "PIL", "PIL.Image", "PIL.ImageTk",
        "sounddevice", "numpy", "scipy", "scipy.signal", "openwakeword",
        "openwakeword.model", "ollama", "ddgs", "ddgs.exceptions",
        "duckduckgo_search", "duckduckgo_search.exceptions",
    )
}
fake_modules["ddgs.exceptions"].DDGSException = SearchError
fake_modules["duckduckgo_search.exceptions"].DuckDuckGoSearchException = SearchError
with mock.patch.dict(sys.modules, fake_modules):
    web_search = importlib.import_module("web_search")
    agent = importlib.import_module("agent")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.MagicMock()
        self.client.__enter__.return_value = self.client
        self.factory = mock.patch.object(web_search, "DDGS", return_value=self.client)
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def test_general_search_preserves_multiple_sources(self):
        self.client.text.return_value = [
            {"title": "First", "href": "https://example.com/1", "body": "One"},
            {"title": "Second", "url": "https://example.com/2", "date": "2026-01-01", "body": "Two"},
        ]
        result = web_search.search_web("robots")
        self.client.news.assert_not_called()
        self.client.text.assert_called_once_with("robots", region="us-en", max_results=3)
        self.assertIn("https://example.com/1", result)
        self.assertIn("Date: 2026-01-01\nURL: https://example.com/2", result)

    def test_news_falls_back_to_general_search(self):
        self.client.news.side_effect = SearchError("provider unavailable")
        self.client.text.return_value = [{"title": "Result", "href": "https://example.com"}]
        self.assertIn("Result", web_search.search_web("robots", news=True))
        self.client.text.assert_called_once()

    def test_empty_and_failure_are_distinct(self):
        self.client.text.side_effect = SearchError("No results found.")
        self.assertEqual(web_search.search_web("robots"), "SEARCH_EMPTY")
        self.client.text.side_effect = SearchError("connection failed")
        self.assertEqual(web_search.search_web("robots"), "SEARCH_ERROR")
        self.assertEqual(web_search.search_web(None), "INVALID_ACTION")

    def test_news_detection(self):
        self.assertTrue(web_search.is_news_query("Latest robot headlines"))
        self.assertFalse(web_search.is_news_query("How do robot sensors work?"))


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.bot = agent.BotGUI.__new__(agent.BotGUI)
        self.bot.permanent_memory = [{"role": "system", "content": "test"}]
        self.bot.session_memory = []
        self.bot.current_state = agent.BotStates.THINKING
        self.bot.thinking_sound_active = threading.Event()
        self.bot.interrupted = threading.Event()
        self.bot.tts_queue = []
        self.bot.tts_queue_lock = threading.Lock()
        self.bot.set_state = mock.Mock()
        self.bot.append_to_text = mock.Mock()
        self.bot._stream_to_text = mock.Mock()
        self.bot.wait_for_tts = mock.Mock()
        self.bot._run_thinking_sound_loop = mock.Mock()
        self.thread = mock.patch.object(agent.threading, "Thread")
        self.thread.start()
        self.addCleanup(self.thread.stop)

    def test_native_tool_search_displays_sources_but_does_not_speak_urls(self):
        tool_response = {"message": {"tool_calls": [
            {"function": {"name": "search_web", "arguments": {"query": "robot news"}}}
        ]}}
        if real_ollama:
            tool_response = real_ollama.ChatResponse(model="qwen3:1.7b", message={
                "role": "assistant", **tool_response["message"]
            })
        search_result = ("SEARCH RESULTS for 'robot news':\n\n"
                         "Title: Robot story\nDate: 2026-01-01\n"
                         "URL: https://example.com/story\nSnippet: Robots.")
        with mock.patch.dict(agent.CURRENT_CONFIG, {"native_tools": True}), \
             mock.patch.object(agent, "search_web", return_value=search_result) as search, \
             mock.patch.object(agent.ollama, "chat", side_effect=[
                 tool_response, {"message": {"content": "Robots are in the news. https://example.com/story"}}
             ]) as chat:
            self.bot.chat_and_respond("What are the robot headlines?")
        search.assert_called_once_with("robot news", news=True)
        self.assertFalse(chat.call_args_list[0].kwargs["think"])
        self.assertTrue(chat.call_args_list[0].kwargs["tools"])
        self.assertFalse(chat.call_args_list[1].kwargs["think"])
        self.assertEqual(self.bot.tts_queue, ["Robots are in the news."])
        self.assertIn("Robot story (2026-01-01): https://example.com/story",
                      str(self.bot.append_to_text.call_args_list))

    def test_legacy_json_split_across_chunks_routes_action(self):
        chunks = [
            {"message": {"content": "{"}},
            {"message": {"content": '"action": "get_time", "value": "now"}'}},
        ]
        with mock.patch.dict(agent.CURRENT_CONFIG, {"native_tools": False}), \
             mock.patch.object(agent.ollama, "chat", side_effect=[
                 iter(chunks), {"message": {"content": "It is noon."}}
             ]), mock.patch.object(self.bot, "execute_action_and_get_result", return_value="The time is noon.") as action:
            self.bot.chat_and_respond("What time is it?")
        self.assertEqual(action.call_args.args[0]["action"], "get_time")
        self.bot._stream_to_text.assert_not_called()
        self.assertEqual(self.bot.tts_queue, ["It is noon."])

    def test_joke_request_does_not_call_model(self):
        with mock.patch.object(agent.ollama, "chat") as chat:
            self.bot.chat_and_respond("Tell me a joke!")
        chat.assert_not_called()
        self.assertIn(self.bot.tts_queue[0], agent.JOKES)

    def test_interrupted_tool_request_does_not_execute(self):
        response = {"message": {"tool_calls": [
            {"function": {"name": "search_web", "arguments": {"query": "robots"}}}
        ]}}
        def interrupt(**kwargs):
            self.bot.interrupted.set()
            return response
        with mock.patch.dict(agent.CURRENT_CONFIG, {"native_tools": True}), \
             mock.patch.object(agent.ollama, "chat", side_effect=interrupt), \
             mock.patch.object(self.bot, "execute_action_and_get_result") as action:
            self.bot.chat_and_respond("Search for robots")
        action.assert_not_called()

    def test_saved_history_uses_current_prompt(self):
        history = [{"role": "system", "content": "old"}, {"role": "assistant", "content": "hello"}]
        with mock.patch.object(agent.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(history))):
            restored = self.bot.load_chat_history()
        self.assertEqual(restored[0]["content"], agent.SYSTEM_PROMPT)
        self.assertEqual(restored[1]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
