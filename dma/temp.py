#
#from __future__ import annotations
#
#import argparse
#from collections import Counter
#from pathlib import Path
#import re
#from typing import Any
#
#from email_validator import EmailNotValidError, validate_email
#import phonenumbers
#import pandas as pd
#import probablepeople
#from stdnum import luhn
#import usaddress
#from gliner import GLiNER

import sys

if len(sys.argv) > 1:
    print(int(sys.argv[1]))
else:
    print("No argument provided")