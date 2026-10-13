"""Tests for the AI-free paired training-data audit."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_data import inspect_sequence


def write_video(path: Path, frame_count: int, fps: float, shift: int) -> bool:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (180, 240)
    )
    if not writer.isOpened():
        writer.release()
        return False
    for index in range(frame_count):
        image = np.full((240, 180, 3), 30, dtype=np.uint8)
        cv2.circle(image, (60 + index + shift, 50), 22, (30, 220, 240), -1)
        cv2.rectangle(image, (48 + index + shift, 72), (72 + index + shift, 170), (200, 70, 30), -1)
        writer.write(image)
    writer.release()
    return True


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sequence = self.root / "sequence_001"
        self.sequence.mkdir()
        self.report = self.root / "report"
        self.report.mkdir()
        reference = np.full((240, 180, 3), 170, dtype=np.uint8)
        cv2.circle(reference, (90, 60), 32, (50, 170, 230), -1)
        cv2.rectangle(reference, (55, 92), (125, 220), (80, 50, 200), -1)
        cv2.imwrite(str(self.sequence / "reference.jpg"), reference)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_pair_creates_contact_sheet(self):
        if not write_video(self.sequence / "driving.mp4", 24, 24.0, 0):
            self.skipTest("MP4V encoder unavailable in this environment")
        if not write_video(self.sequence / "target.mp4", 24, 24.0, 8):
            self.skipTest("MP4V encoder unavailable in this environment")
        result = inspect_sequence(self.sequence, self.report)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(Path(result["contact_sheet"]).is_file())

    def test_slightly_different_frame_counts_are_allowed_with_warning(self):
        if not write_video(self.sequence / "driving.mp4", 24, 24.0, 0):
            self.skipTest("MP4V encoder unavailable in this environment")
        if not write_video(self.sequence / "target.mp4", 20, 24.0, 8):
            self.skipTest("MP4V encoder unavailable in this environment")
        result = inspect_sequence(self.sequence, self.report)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(any("Frame counts differ" in warning for warning in result["warnings"]))

    def test_identical_source_and_target_are_rejected(self):
        source = self.sequence / "driving.mp4"
        if not write_video(source, 24, 24.0, 0):
            self.skipTest("MP4V encoder unavailable in this environment")
        (self.sequence / "target.mp4").write_bytes(source.read_bytes())
        result = inspect_sequence(self.sequence, self.report)
        self.assertFalse(result["ok"])
        self.assertTrue(any("byte-for-byte identical" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
