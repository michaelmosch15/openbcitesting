"""Cyton packet parsing with synthetic 33-byte packets (no serial port needed)."""
import threading
import unittest

import numpy as np

import _path  # noqa: F401
import cyton
from cyton import PACKET, UV_PER_COUNT, ACCEL_G_PER_COUNT, CytonBoard, SampleBuffer


def int24(v: int) -> bytes:
    return int(v).to_bytes(3, "big", signed=True)


def make_packet(idx: int, counts, accel=(0, 0, 0), stop=0xC0) -> bytes:
    """A0 | sample# | 8 x int24 BE | 3 x int16 BE | C0..CF  (33 bytes)."""
    assert len(counts) == 8
    body = b"".join(int24(c) for c in counts)
    body += b"".join(int(a).to_bytes(2, "big", signed=True) for a in accel)
    pkt = bytes([0xA0, idx & 0xFF]) + body + bytes([stop])
    assert len(pkt) == PACKET
    return pkt


class FakeSerial:
    """read() hands out queued chunks; once drained it sets the board's stop event so
    `_reader` returns. `read` ignores the requested size on purpose."""

    def __init__(self, chunks, stop_event):
        self.chunks = list(chunks)
        self.stop = stop_event
        self.written = b""

    def read(self, n=1):
        if self.chunks:
            return self.chunks.pop(0)
        self.stop.set()
        return b""

    def write(self, b):
        self.written += b

    def flush(self):
        pass


def make_board(chunks, n_ch=8, daisy_even_is_upper=True):
    b = CytonBoard.__new__(CytonBoard)
    b._stop = threading.Event()
    b.ser = FakeSerial(chunks, b._stop)
    b.n_ch = n_ch
    b.fs = 125.0 if n_ch == 16 else 250.0
    b.daisy_even_is_upper = daisy_even_is_upper
    b.buffer = SampleBuffer(n_ch, b.fs, 2.0)
    b.packets = 0
    b.dropped = 0
    b.last_packet_t = 0.0
    return b


class TestScale(unittest.TestCase):
    def test_uv_per_count(self):
        self.assertAlmostEqual(UV_PER_COUNT, 0.02235, places=5)
        self.assertAlmostEqual(UV_PER_COUNT, 4.5 / 24.0 / (2 ** 23 - 1) * 1e6, places=12)

    def test_full_scale(self):
        self.assertAlmostEqual(cyton.FULL_SCALE_UV, 187500.0, delta=1.0)

    def test_int24_sign_handling(self):
        cases = {b"\x00\x00\x01": 1, b"\xff\xff\xff": -1, b"\x7f\xff\xff": 8388607,
                 b"\x80\x00\x00": -8388608, b"\x00\x00\x00": 0}
        for raw, expect in cases.items():
            v = int.from_bytes(raw, "big", signed=True)
            self.assertEqual(v, expect)
            self.assertAlmostEqual(v * UV_PER_COUNT, expect * UV_PER_COUNT)
        # the encoder used by the tests round-trips through the same expression
        for v in (0, 1, -1, 123456, -123456, 8388607, -8388608):
            self.assertEqual(int.from_bytes(int24(v), "big", signed=True), v)


class TestSampleBuffer(unittest.TestCase):
    def test_push_last_order_and_wraparound(self):
        buf = SampleBuffer(2, 10.0, 1.0)          # size 10
        self.assertEqual(buf.size, 10)
        x = np.arange(14, dtype=np.float32)
        eeg = np.stack([x, -x], axis=1)
        acc = np.zeros((14, 3), np.float32)
        t = 100.0 + np.arange(14) / 10.0
        buf.push(eeg[:7], acc[:7], t[:7])
        self.assertEqual(buf.count, 7)
        d, tt = buf.last(3)
        np.testing.assert_array_equal(d[:, 0], [4, 5, 6])
        np.testing.assert_allclose(tt, t[4:7])
        buf.push(eeg[7:], acc[7:], t[7:])          # wraps past the end of the ring
        self.assertEqual(buf.count, 14)
        self.assertEqual(buf.pos, 4)
        d, tt = buf.last(10)
        np.testing.assert_array_equal(d[:, 0], np.arange(4, 14))
        np.testing.assert_array_equal(d[:, 1], -np.arange(4, 14))
        np.testing.assert_allclose(tt, t[4:14])
        # asking for more than the ring holds caps at the ring size
        d, _ = buf.last(50)
        self.assertEqual(len(d), 10)

    def test_last_when_empty(self):
        buf = SampleBuffer(3, 250.0, 1.0)
        d, t = buf.last(5)
        self.assertEqual(d.shape, (0, 3))
        self.assertEqual(len(t), 0)

    def test_listener_gets_count0(self):
        buf = SampleBuffer(1, 10.0, 1.0)
        seen = []
        buf.listeners.append(lambda e, a, t, c0: seen.append((len(e), c0)))
        buf.push(np.zeros((3, 1)), np.zeros((3, 3)), np.zeros(3))
        buf.push(np.zeros((2, 1)), np.zeros((2, 3)), np.zeros(2))
        self.assertEqual(seen, [(3, 0), (2, 3)])


class TestReader(unittest.TestCase):
    def test_clean_8ch_packet(self):
        counts = [1000, -1000, 0, 8388607, -8388608, 1, -1, 44739]
        b = make_board([make_packet(7, counts, accel=(16, -16, 8000))])
        b._reader()
        self.assertEqual(b.packets, 1)
        self.assertEqual(b.dropped, 0)
        self.assertEqual(b.buffer.count, 1)
        d, t = b.buffer.last(1)
        np.testing.assert_allclose(d[0], np.array(counts) * UV_PER_COUNT, rtol=1e-5)
        self.assertGreater(t[0], 0)
        self.assertGreater(b.last_packet_t, 0)
        np.testing.assert_allclose(b.buffer.accel[0], np.array([16, -16, 8000]) * ACCEL_G_PER_COUNT, rtol=1e-5)

    def test_garbage_before_packet_is_skipped(self):
        counts = [10, 20, 30, 40, 50, 60, 70, 80]
        junk = b"\x12\x34\xa0\x00"       # includes a stray A0 that is not a packet start
        b = make_board([junk + make_packet(1, counts)])
        b._reader()
        self.assertEqual(b.packets, 1)
        d, _ = b.buffer.last(1)
        np.testing.assert_allclose(d[0], np.array(counts) * UV_PER_COUNT, rtol=1e-5)

    def test_packet_split_across_reads(self):
        pkt = make_packet(3, [1, 2, 3, 4, 5, 6, 7, 8], stop=0xC3)
        b = make_board([pkt[:10], pkt[10:20], pkt[20:]])
        b._reader()
        self.assertEqual(b.packets, 1)
        d, _ = b.buffer.last(1)
        np.testing.assert_allclose(d[0], np.arange(1, 9) * UV_PER_COUNT, rtol=1e-5)

    def test_sample_index_gap_counts_dropped(self):
        z = [0] * 8
        b = make_board([make_packet(5, z) + make_packet(6, z) + make_packet(9, z)])
        b._reader()
        self.assertEqual(b.packets, 3)
        self.assertEqual(b.dropped, 2)

    def test_index_wraparound_is_not_a_drop(self):
        z = [0] * 8
        b = make_board([make_packet(254, z) + make_packet(255, z) + make_packet(0, z) + make_packet(1, z)])
        b._reader()
        self.assertEqual(b.packets, 4)
        self.assertEqual(b.dropped, 0)

    def test_16ch_combines_odd_lower_with_even_upper(self):
        odd = [100, 200, 300, 400, 500, 600, 700, 800]        # channels 1-8 (Cyton)
        even = [-100, -200, -300, -400, -500, -600, -700, -800]  # channels 9-16 (Daisy)
        b = make_board([make_packet(1, odd), make_packet(2, even)], n_ch=16, daisy_even_is_upper=True)
        b._reader()
        self.assertEqual(b.packets, 2)
        self.assertEqual(b.dropped, 0)
        self.assertEqual(b.buffer.count, 1)
        self.assertEqual(b.buffer.n_ch, 16)
        self.assertEqual(b.fs, 125.0)
        d, _ = b.buffer.last(1)
        self.assertEqual(d.shape, (1, 16))
        np.testing.assert_allclose(d[0, :8], np.array(odd) * UV_PER_COUNT, rtol=1e-5)
        np.testing.assert_allclose(d[0, 8:], np.array(even) * UV_PER_COUNT, rtol=1e-5)

    def test_16ch_even_first_pairs_with_following_odd(self):
        odd = [1] * 8
        even = [2] * 8
        # even packet first (BrainFlow behaviour): it pairs with the next odd packet; the
        # following odd/even pair gives a second sample
        chunks = [make_packet(0, even), make_packet(1, odd), make_packet(3, odd), make_packet(4, even)]
        b = make_board(chunks, n_ch=16, daisy_even_is_upper=True)
        b._reader()
        self.assertEqual(b.packets, 4)
        self.assertEqual(b.dropped, 1)          # idx 2 missing
        self.assertEqual(b.buffer.count, 2)
        d, t = b.buffer.last(2)
        for row in d:
            np.testing.assert_allclose(row[:8], UV_PER_COUNT, rtol=1e-5)
            np.testing.assert_allclose(row[8:], 2 * UV_PER_COUNT, rtol=1e-5)
        # one 16-ch sample per two packets at 250 pkt/s -> 8 ms apart
        self.assertAlmostEqual(t[1] - t[0], 2.0 / 250.0, delta=1e-3)

    def test_16ch_same_parity_twice_keeps_newest_half(self):
        b = make_board([make_packet(1, [1] * 8), make_packet(3, [3] * 8), make_packet(4, [4] * 8)],
                       n_ch=16, daisy_even_is_upper=True)
        b._reader()
        self.assertEqual(b.buffer.count, 1)
        d, _ = b.buffer.last(1)
        np.testing.assert_allclose(d[0, :8], 3 * UV_PER_COUNT, rtol=1e-5)
        np.testing.assert_allclose(d[0, 8:], 4 * UV_PER_COUNT, rtol=1e-5)

    def test_16ch_flip_flag_swaps_halves(self):
        odd = [1] * 8
        even = [2] * 8
        b = make_board([make_packet(1, odd) + make_packet(2, even)], n_ch=16, daisy_even_is_upper=False)
        b._reader()
        self.assertEqual(b.buffer.count, 1)
        d, _ = b.buffer.last(1)
        # flipped: odd packets fill channels 9-16, even packets channels 1-8
        np.testing.assert_allclose(d[0, :8], 2 * UV_PER_COUNT, rtol=1e-5)
        np.testing.assert_allclose(d[0, 8:], UV_PER_COUNT, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
