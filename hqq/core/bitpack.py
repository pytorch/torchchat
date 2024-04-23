# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################

import torch
from torch import uint8, int32, Tensor
import numpy as np
from .utils import is_divisible


# Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
    # 8-bit
    ################################################
    @staticmethod
    def pack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q.to(uint8)

    @staticmethod
    def unpack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q

    # 4-bit
    ################################################
    @staticmethod
    def pack_4bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/2
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 2)

        return (W_q[:_step] << 4) | W_q[_step:]

    @staticmethod
    def unpack_4bit_u8_cat(W_q: Tensor) -> Tensor:  # uint8/2 > uint8
        return torch.cat([(W_q & 0b11110000) >> 4, W_q & 0b00001111], axis=0)

    # A bit faster than the _cat version
    @staticmethod
    def unpack_4bit_u8(W_q: Tensor) -> Tensor:  # uint8/2 > uint8
        _step = W_q.shape[0]
        tmp = torch.empty([2 * _step, W_q.shape[1]], dtype=uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b11110000) >> 4
        tmp[_step:] = W_q & 0b00001111

        return tmp

    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 4)

        return (
            W_q[:_step] << 6
            | W_q[_step : 2 * _step] << 4
            | W_q[2 * _step : 3 * _step] << 2
            | W_q[3 * _step :]
        )

    @staticmethod
    def unpack_2bit_u8_cat(W_q: Tensor) -> Tensor:
        return torch.cat(
            [
                (W_q & 0b11000000) >> 6,
                (W_q & 0b00110000) >> 4,
                (W_q & 0b00001100) >> 2,
                W_q & 0b00000011,
            ],
            axis=0,
        )

    # A bit faster than the _cat version
    @staticmethod
    def unpack_2bit_u8(W_q: Tensor) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([4 * _step, W_q.shape[1]], dtype=uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b11000000) >> 6
        tmp[_step : 2 * _step] = (W_q & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (W_q & 0b00001100) >> 2
        tmp[3 * _step :] = W_q & 0b00000011

        return tmp

    # 3bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in: Tensor) -> Tensor:
        W_q = torch.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=int32,
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)
        W_q = (
            (W_q[:_step] << 27)
            | (W_q[_step : _step * 2] << 24)
            | (W_q[_step * 2 : _step * 3] << 21)
            | (W_q[_step * 3 : _step * 4] << 18)
            | (W_q[_step * 4 : _step * 5] << 15)
            | (W_q[_step * 5 : _step * 6] << 12)
            | (W_q[_step * 6 : _step * 7] << 9)
            | (W_q[7 * _step : _step * 8] << 6)
            | (W_q[_step * 8 : _step * 9] << 3)
            | (W_q[_step * 9 :])
        )

        return W_q

    @staticmethod
    def unpack_3bit_32_cat(W_q: Tensor) -> Tensor:
        return torch.cat(
            [
                ((W_q & 0b00111000000000000000000000000000) >> 27),
                ((W_q & 0b00000111000000000000000000000000) >> 24),
                ((W_q & 0b00000000111000000000000000000000) >> 21),
                ((W_q & 0b00000000000111000000000000000000) >> 18),
                ((W_q & 0b00000000000000111000000000000000) >> 15),
                ((W_q & 0b00000000000000000111000000000000) >> 12),
                ((W_q & 0b00000000000000000000111000000000) >> 9),
                ((W_q & 0b00000000000000000000000111000000) >> 6),
                ((W_q & 0b00000000000000000000000000111000) >> 3),
                (W_q & 0b00000000000000000000000000000111),
            ],
            axis=0,
        )

    # A bit faster than _cat version
    @staticmethod
    def unpack_3bit_32(W_q: Tensor) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b00111000000000000000000000000000) >> 27
        tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
        tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
        tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
        tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
        tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
        tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
        tmp[9 * _step :] = W_q & 0b00000000000000000000000000000111

        return tmp

    # 1bit
    ################################################
    @staticmethod
    def pack_1bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 8)

        return (
            W_q[:_step] << 7
            | W_q[_step : 2 * _step] << 6
            | W_q[2 * _step : 3 * _step] << 5
            | W_q[3 * _step : 4 * _step] << 4
            | W_q[4 * _step : 5 * _step] << 3
            | W_q[5 * _step : 6 * _step] << 2
            | W_q[6 * _step : 7 * _step] << 1
            | W_q[7 * _step :]
        )

    @staticmethod
    def unpack_1bit_u8(W_q: Tensor) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([8 * _step, W_q.shape[1]], dtype=uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b10000000) >> 7

        tmp[1 * _step : 2 * _step] = (W_q & 0b01000000) >> 6
        tmp[2 * _step : 3 * _step] = (W_q & 0b00100000) >> 5
        tmp[3 * _step : 4 * _step] = (W_q & 0b00010000) >> 4
        tmp[4 * _step : 5 * _step] = (W_q & 0b00001000) >> 3
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000100) >> 2
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000010) >> 1
        tmp[7 * _step : 8 * _step] = W_q & 0b00000001

        return tmp

    # Experimental
    ################################################################################################################
    @staticmethod
    def pack_3bit2bit_u8(W_q: Tensor) -> Tensor:
        assert is_divisible(
            len(W_q), 3
        ), "Input should have shape[0] divisble by 3 to use mixed 3-2bit bit packing"
        _step = int(len(W_q) / 3)

        return W_q[:_step] << 6 | W_q[1 * _step : 2 * _step] << 3 | W_q[2 * _step :]

    @staticmethod
    def unpack_3bit2bit_u8(W_q: Tensor) -> Tensor:
        return torch.cat(
            [(W_q & 0b11100000) >> 6, (W_q & 0b00011100) >> 3, W_q & 0b00000011], axis=0
        )

    @staticmethod
    def pack_4bit_32(W_q: Tensor) -> Tensor:
        W_q = W_q.to(int32)
        _step = int(len(W_q) / 8)
        W_q = (
            (W_q[:_step] << 28)
            | (W_q[_step : _step * 2] << 24)
            | (W_q[_step * 2 : _step * 3] << 20)
            | (W_q[_step * 3 : _step * 4] << 16)
            | (W_q[_step * 4 : _step * 5] << 12)
            | (W_q[_step * 5 : _step * 6] << 8)
            | (W_q[_step * 6 : _step * 7] << 4)
            | (W_q[_step * 7 :])
        )
        return W_q

    @staticmethod
    def unpack_4bit_32(W_q: Tensor) -> Tensor:
        return torch.cat(
            [
                ((W_q & 0b11110000000000000000000000000000) >> 28),
                ((W_q & 0b00001111000000000000000000000000) >> 24),
                ((W_q & 0b00000000111100000000000000000000) >> 20),
                ((W_q & 0b00000000000011110000000000000000) >> 16),
                ((W_q & 0b00000000000000001111000000000000) >> 12),
                ((W_q & 0b00000000000000000000111100000000) >> 8),
                ((W_q & 0b00000000000000000000000011110000) >> 4),
                (W_q & 0b00000000000000000000000000001111),
            ],
            axis=0,
        )
