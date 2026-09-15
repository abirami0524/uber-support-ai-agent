import unittest
from unittest.mock import patch

import explore


class ExploreDatasetTest(unittest.TestCase):
    @patch("explore.kagglehub.dataset_download", return_value="/tmp/customer-support-on-twitter")
    def test_download_dataset_returns_dataset_path(self, mock_download):
        dataset_id = "thoughtvector/customer-support-on-twitter"

        result = explore.download_dataset(dataset_id)

        self.assertEqual(result, "/tmp/customer-support-on-twitter")
        mock_download.assert_called_once_with(dataset_id)


if __name__ == "__main__":
    unittest.main()
