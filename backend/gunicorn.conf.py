import os

timeout = 120
workers = 1
threads = 4
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"