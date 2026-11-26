# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pickle
import struct
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import skhep_testdata
import xxhash

import uproot


def _create_rntuple_with_attribute_set(path: Path) -> Path:
    """
    Creates an RNTuple file with a main RNTuple and an attribute set linked
    in the footer with schema version 1.0.
    """
    with uproot.recreate(path) as f:
        f.mkrntuple("main", {"x": np.arange(25, dtype=np.int32)})
        f.mkrntuple(
            "attr",
            {
                "_rangeStart": np.array([0, 10], dtype=np.uint64),
                "_rangeLen": np.array([10, 15], dtype=np.uint64),
                "_userData": ak.Array(
                    [
                        {"note": "run_a", "energy": 13.6},
                        {"note": "run_b", "energy": 14.0},
                    ]
                ),
            },
        )

    with uproot.open(path) as f:
        main_key = f.key("main")
        attr_key = f.key("attr")
        attr_anchor_seek = attr_key.data_cursor.index + 6
        main_anchor_seek = main_key.data_cursor.index + 6

    name = b"run_conditions"
    attr_record = (
        struct.pack("<HHI", 1, 0, 72)
        + struct.pack("<iQ", 72, attr_anchor_seek)
        + struct.pack("<I", len(name))
        + name
    )
    record_frame = struct.pack("<q", len(attr_record) + 8) + attr_record
    list_frame = struct.pack("<qI", -(len(record_frame) + 12), 1) + record_frame

    raw = bytearray(path.read_bytes())
    anchor_bytes = bytearray(raw[main_anchor_seek : main_anchor_seek + 72])
    struct.pack_into(">H", anchor_bytes, 4, 1)

    with uproot.open(path) as f:
        f_seek = f["main"].member("fSeekFooter")
        f_len = f["main"].member("fLenFooter")

    old_footer_bytes = bytes(raw[f_seek : f_seek + f_len])
    body_before_attr = old_footer_bytes[16:-8]
    new_body = body_before_attr + list_frame

    new_footer_len = 16 + len(new_body) + 8
    env_header = struct.pack("<H6sQ", 2, b"\x00" * 6, new_footer_len)
    footer_without_cksum = env_header + new_body
    footer_cksum = xxhash.xxh3_64_intdigest(footer_without_cksum)
    new_footer = footer_without_cksum + struct.pack("<Q", footer_cksum)

    new_footer_seek = len(raw)
    raw.extend(new_footer)

    struct.pack_into(
        ">QQQ", anchor_bytes, 32, new_footer_seek, new_footer_len, new_footer_len
    )
    new_anchor_cksum = xxhash.xxh3_64_intdigest(bytes(anchor_bytes[:64]))
    anchor_bytes[64:72] = struct.pack(">Q", new_anchor_cksum)
    raw[main_anchor_seek : main_anchor_seek + 72] = anchor_bytes

    path.write_bytes(raw)
    return path


def test_rntuple_attributes_full_lifecycle(tmp_path):
    path = _create_rntuple_with_attribute_set(tmp_path / "attrs.root")
    with uproot.open(path) as f:
        main = f["main"]

        assert "run_conditions" in main.attribute_sets
        assert len(main.attribute_sets) == 1
        assert list(main.attribute_sets.keys()) == ["run_conditions"]

        attr = main.attribute_set("run_conditions")
        assert attr.name == "run_conditions"
        assert attr.schema_version == (1, 0)
        assert attr.num_entries == 2
        assert len(attr) == 2
        assert list(attr.keys()) == ["note", "energy"]
        assert "note" in attr
        assert "energy" in attr
        assert "_userData" in attr
        assert "_rangeStart" in attr

        assert attr["note"].array().to_list() == ["run_a", "run_b"]
        assert attr["energy"].array().to_list() == [13.6, 14.0]

        assert attr.ranges == [(0, 10), (10, 25)]
        assert attr.range_start.tolist() == [0, 10]
        assert attr.range_len.tolist() == [10, 15]
        assert attr.range_end.tolist() == [10, 25]

        assert attr.locate_range(0) == 0
        assert attr.locate_range(5) == 0
        assert attr.locate_range(9) == 0
        assert attr.locate_range(10) == 1
        assert attr.locate_range(20) == 1
        assert attr.locate_range(24) == 1
        assert attr.locate_range(25) is None
        assert attr.locate_range(-1) is None
        assert attr.locate_range(100) is None

        row0 = attr.for_entry(5)
        assert row0 is not None
        assert row0.index == 0
        assert row0.range == (0, 10)
        assert row0.range_start == 0
        assert row0.range_end == 10
        assert row0["note"] == "run_a"
        assert float(row0["energy"]) == 13.6
        assert row0.to_dict()["note"] == "run_a"
        assert float(row0.to_dict()["energy"]) == 13.6
        assert list(row0.keys()) == ["note", "energy"]
        assert len(row0) == 2
        assert "note" in row0
        assert list(row0) == ["note", "energy"]

        row1 = attr.for_entry(15)
        assert row1 is not None
        assert row1.index == 1
        assert row1.range == (10, 25)
        assert row1.range_start == 10
        assert row1.range_end == 25
        assert row1["note"] == "run_b"
        assert float(row1["energy"]) == 14.0
        assert row1.to_dict()["note"] == "run_b"
        assert float(row1.to_dict()["energy"]) == 14.0

        assert attr.for_entry(25) is None
        assert attr.for_entry(25, default="missing") == "missing"
        assert attr.for_entry(-1, default=None) is None

        arr = attr.arrays()
        assert arr.note.to_list() == ["run_a", "run_b"]
        assert arr.energy.to_list() == [13.6, 14.0]

        arr_note = attr.arrays(["note"])
        assert arr_note.note.to_list() == ["run_a", "run_b"]

        repr_attr = repr(attr)
        assert "<RNTupleAttributeSet 'run_conditions'" in repr_attr
        assert "2 attributes" in repr_attr
        assert "2 entries" in repr_attr

        repr_row = repr(row0)
        assert "<RNTupleAttributeRow 'run_conditions'" in repr_row
        assert "[0, 10)" in repr_row
        assert "entry=0" in repr_row

        with pytest.raises(KeyError, match="has no attribute set 'nonexistent'"):
            main.attribute_set("nonexistent")

        with pytest.raises(KeyError, match="not found in attribute set"):
            _ = attr["nonexistent"]


def test_rntuple_attributes_empty_in_v1010():
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-1-0.root")
    with uproot.open(filename) as f:
        obj = f["Staff"]
        assert obj.attribute_sets == {}
        with pytest.raises(KeyError, match="has no attribute set 'something'"):
            obj.attribute_set("something")


def test_rntuple_attributes_empty_in_v1000():
    filename = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["Staff"]
        assert obj.attribute_sets == {}
        with pytest.raises(KeyError, match="has no attribute set 'something'"):
            obj.attribute_set("something")


def test_rntuple_attributes_pickle(tmp_path):
    path = _create_rntuple_with_attribute_set(tmp_path / "attrs_pickle.root")
    with uproot.open(path) as f:
        main = f["main"]
        attr = main.attribute_set("run_conditions")
        assert attr.ranges == [(0, 10), (10, 25)]

        unpickled_main = pickle.loads(pickle.dumps(main))
        assert "run_conditions" in unpickled_main.attribute_sets
        unpickled_attr = unpickled_main.attribute_set("run_conditions")
        assert unpickled_attr.ranges == [(0, 10), (10, 25)]
        assert unpickled_attr["note"].array().to_list() == ["run_a", "run_b"]


def test_rntuple_attributes_show(tmp_path):
    import io

    path = _create_rntuple_with_attribute_set(tmp_path / "attrs_show.root")
    with uproot.open(path) as f:
        attr = f["main"].attribute_set("run_conditions")
        s = io.StringIO()
        attr.show(stream=s)
        out = s.getvalue()
        assert "note" in out
        assert "energy" in out
