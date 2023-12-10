#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@version:0.1
@author:Cai Qingpeng
@file: arguments.py
@time: 2020/11/30 1:42 PM
'''


import argparse

def Dipole_parse_args():
    parser = argparse.ArgumentParser(description="Run Dipole on regular dataset.",add_help=False)

    # model
    parser.add_argument('--model', type=str,required=True, choices=["Dip_l","Dip_g","Dip_c"],
                        help="model")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="hidden_dim")
    parser.add_argument('--bi_direction', action='store_true', default=True, # Dipole uses bi-direction GRU
                        help="bi_direction")

    return parser