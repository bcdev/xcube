# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest


from xcube.core.store import PreloadStatus
from xcube.core.store import PreloadState


class PreloadStateTest(unittest.TestCase):
    def test_str(self):
        state = PreloadState(
            "test.zip", status=PreloadStatus.started, progress=0.71, message="Unzipping"
        )
        self.assertEqual(
            "data_id=test.zip, "
            "status=STARTED, "
            "progress=0.71, "
            "message=Unzipping",
            str(state),
        )

    def test_repr(self):
        state = PreloadState(
            "test.zip", status=PreloadStatus.started, progress=0.71, message="Unzipping"
        )
        self.assertEqual(
            "PreloadState("
            "data_id='test.zip', "
            "status=PreloadStatus.started, "
            "progress=0.71, "
            "message='Unzipping')",
            repr(state),
        )


class PreloadStatusTest(unittest.TestCase):
    def test_str(self):
        self.assertEqual("CANCELLED", str(PreloadStatus.cancelled))

    def test_repr(self):
        self.assertEqual("PreloadStatus.started", repr(PreloadStatus.started))
