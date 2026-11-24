#!/usr/bin/env python3
"""ELF machine type detector with EM_CHILI support."""
import struct
import sys
from pathlib import Path

EM_MAP = {
    0: "EM_NONE",
    3: "EM_386",
    40: "EM_ARM",
    62: "EM_X86_64",
    183: "EM_AARCH64",
    243: "EM_RISCV",
    258: "EM_LOONGARCH",
    272: "EM_CHILI (OpenAI Chili architecture)",
}

def parse_elf_machine(path):
    with open(path, "rb") as f:
        data = f.read(64)
    if data[:4] != b"\x7fELF":
        return None, "Not an ELF"
    ei_class = data[4]
    if ei_class == 2:  # ELF64
        e_machine = struct.unpack_from("<H", data, 18)[0]
    elif ei_class == 1:  # ELF32
        e_machine = struct.unpack_from("<H", data, 18)[0]
    else:
        return None, "Unknown class"
    name = EM_MAP.get(e_machine, f"Unknown (0x{e_machine:x})")
    return e_machine, name

if __name__ == "__main__":
    paths = sys.argv[1:] or ["/workspace/artifacts/chili_test.elf"]
    print(f"{'File':<40} {'e_machine':>10}  Name")
    print("-" * 70)
    for p in paths:
        em, name = parse_elf_machine(p)
        print(f"{Path(p).name:<40} {em:>10}  {name}")
