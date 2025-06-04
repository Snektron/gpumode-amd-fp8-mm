#!/usr/bin/env python3
import base64
import zlib

with open('solution.hip', 'rb') as f:
	hip = f.read()

with open('solution.template.py') as f:
	template = f.read()

with open('solution.py', 'w') as f:
	f.write(template.replace('@SOLUTION@', base64.b64encode(zlib.compress(hip)).decode()))
