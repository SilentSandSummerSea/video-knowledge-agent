import unittest
from datetime import datetime, timedelta, timezone

from video_knowledge_agent.costing import estimate_deepseek_flash_cost, pricing_period


class CostingTests(unittest.TestCase):
    def test_off_peak_cost_does_not_double_count_reasoning(self):
        estimate = estimate_deepseek_flash_cost(
            {
                "input_tokens": 100_000,
                "cached_input_tokens": 20_000,
                "output_tokens": 200_000,
                "reasoning_tokens": 150_000,
            },
            period="off_peak",
        )
        self.assertEqual(estimate["estimated_cost_cny"], 0.8804)

    def test_weekday_peak_and_weekend_off_peak(self):
        zone = timezone(timedelta(hours=8))
        self.assertEqual(pricing_period(datetime(2026, 9, 28, 10, tzinfo=zone)), "peak")
        self.assertEqual(pricing_period(datetime(2026, 9, 27, 10, tzinfo=zone)), "off_peak")


if __name__ == "__main__":
    unittest.main()
