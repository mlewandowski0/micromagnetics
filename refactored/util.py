from time import time

def format_time(starting_time):
    v = round(time() - starting_time, 3)
    return f"{ v//3600}h {(v % 3600 )//60}m {round(v % 60, 3)}s"
