"""Unit tests for route decisions and bounded retrieval retries."""

import unittest

from src.tools.graph_tools import doc_tool, routing_tool


class RoutingTests(unittest.TestCase):
    def test_index_route(self):
        self.assertEqual(routing_tool({"route": "index"}), "retriever")

    def test_general_route_skips_retrieval(self):
        self.assertEqual(routing_tool({"route": "general"}), "general_llm")

    def test_search_route(self):
        self.assertEqual(routing_tool({"route": "search"}), "web_search")

    def test_relevant_context_goes_to_generation(self):
        self.assertEqual(
            doc_tool({"binary_score": "yes", "retrieval_attempts": 1}),
            "generate",
        )

    def test_failed_retrieval_retries_before_limit(self):
        self.assertEqual(
            doc_tool({"binary_score": "no", "retrieval_attempts": 2}),
            "rewrite",
        )

    def test_failed_retrieval_falls_back_at_limit(self):
        self.assertEqual(
            doc_tool({"binary_score": "no", "retrieval_attempts": 3}),
            "web_search",
        )


if __name__ == "__main__":
    unittest.main()
