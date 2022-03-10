"""
文本样例：你好，请问eap202的带机数是多少
标注样例：[[(5, 10), 设备名称], [(12, 14), 设备参数], [参数咨询]]
"""
from __future__ import annotations
from enum import IntEnum
import paddle
import os
import ast
import argparse
import warnings
import numpy as np
import pandas as pd

input_text = '你好，请问eap202的带机数是多少'
input_ano = [[(5, 10), "设备名称"], [(12, 14), "设备参数"], ["参数咨询"]]

def convert_data(input_text, input_ano):
    BIO_list = input_ano[:-1]
    intent_ano = input_ano[-1]
    BIO_ano = ["O"] * len(input_text)
    for li in BIO_list:
        tu = li[0]
        ano_B = "B+Inform+" + li[1]
        ano_I = "I+Inform+" + li[1]
        BIO_ano[tu[0]] = ano_B
        for i in range(tu[0] + 1, tu[1] + 1):
            BIO_ano[i] = ano_I

    return BIO_ano, intent_ano

BIO_ano, intent_ano = convert_data(input_text, input_ano)
print(BIO_ano)
print(intent_ano)