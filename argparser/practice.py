import argparse

parser1 = argparse.ArgumentParser(description="this is a test program made by Naichen")
group = parser1.add_mutually_exclusive_group()
group.add_argument("-f","--foo",help="here is the usage of -f arg",action='count')
group.add_argument("-b","--bar",help="here is the usage of -b arg",action='count')
group.add_argument("-c","--cat",help="here is the usage of -c arg",action='count')
parser1.add_argument("-v","--version",help="show the version of this program",action="store_true")
args = parser1.parse_args()

print(f"args.foo={args.foo}, args.bar={args.bar}, args.cat={args.cat}")
