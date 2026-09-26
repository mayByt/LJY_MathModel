import sys
from .redesign import FIGS

def main():
    args=sys.argv[1:]
    if not args:raise SystemExit('指定图号，例如 Q3-05')
    for arg in args:FIGS[int(arg.split('-')[-1])]()
if __name__=='__main__':main()
