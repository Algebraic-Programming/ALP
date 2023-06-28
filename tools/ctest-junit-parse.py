#
#   Copyright 2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

### starting from the JUnit files produced from CTest, it summarizes test results
# per outcome, printing which tests failed; if requested, remove the logs of successful
# tests in order to keep the size of the artifacts uploaded from the CI minimal
# (by uploading only the logs of failed tests); it parsers the JUnit XML file
# using Python internal 'xml' package

import sys
import os
import argparse
import xml.etree.ElementTree as ET

ALL_CATEGORIES=[ "unit", "performance", "smoke" ]

parser = argparse.ArgumentParser(description='ALP/GraphBLAS test result parser')
parser.add_argument('--categories', help='Test categories', required=True, nargs='+', choices=ALL_CATEGORIES )
parser.add_argument('--xmls-dir',
	help='directory with XML files (one per test category: <category 1>.xml, <category 2>.xml)',
	required=True
)
parser.add_argument('--remove-successful-logs-from',
	help='Remove logs of successful tests stored in this directory, under <this argument>/<category>/output'
)
args = parser.parse_args()

# manipulate an XML nodes to find test names matching a status string
def get_names_list( tags ):
	return [ at.attrib['name'] for at in tags ]

def get_tags_with_status( root, status_str ):
	return root.findall('./testcase[@status="' + status_str + '"]')

def filter_test_names( root, status_str ):
	return [ at.attrib['name'] for at in get_tags_with_status( root, status_str ) ]

# list all tests under the given caption
def list_tests( test_names, caption ):
	if len(test_names) > 0:
		print( caption )
		for tn in test_names:
			print("-", tn)
		print()

# analyze tests for the given category, returning the number of failed ones
# and the list of successful ones
def analyze_tests( xmls_dir, category ):
	xml_path = os.path.join( os.path.abspath( xmls_dir), category + '.xml' )
	tree = ET.parse( xml_path )
	root = tree.getroot()

	# extract counts from root attributes
	at = root.attrib
	num_tests = int( at['tests'] )
	num_failures = int( at['failures'] )
	num_disabled = int( at['disabled'] )
	num_skipped = int( at['skipped'] )
	num_passed = num_tests - num_failures - num_disabled - num_skipped

	# pretty print summary
	lines = [
		( "TESTS:", num_tests ),
		( "success:", num_passed ),
		( "disabled:", num_disabled ),
		( "skipped:", num_skipped ),
		( "FAILED:", num_failures )
	]
	for s, v in lines:
		print( "{:<9} {:>4}".format( s, v ) )
	print()

	# filter test names by status and list them
	if num_disabled > 0:
		disabled_tests = filter_test_names(root, 'disabled')
		list_tests(disabled_tests, "DISABLED TESTS:")
	if num_skipped > 0:
		skipped_tests = filter_test_names(root, 'notrun')
		list_tests(skipped_tests, "SKIPPED TESTS:")
	if num_failures > 0:
		failed_tests = filter_test_names(root, 'fail')
		list_tests(failed_tests, "FAILED TESTS:")
	if num_passed > 0:
		passed_tests = filter_test_names(root, 'run')
	else:
		passed_tests = []
	return num_failures, passed_tests

total_num_failures = 0
per_category_passed_tests = dict()
for category in args.categories:
	print("CATEGORY:", category)
	# for each category, accumulate number of failed tests and store passed tests into a dictionary
	test_num_failures, passed_tests = analyze_tests( args.xmls_dir, category )
	total_num_failures += test_num_failures
	per_category_passed_tests[category] = passed_tests

if args.remove_successful_logs_from is None:
	sys.exit( 0 if total_num_failures == 0 else 1 )

base_directory = os.path.abspath( args.remove_successful_logs_from )
if not os.path.isdir(base_directory):
	print(f"directory {base_directory} does not exist")
	sys.exit(1)

print("removing logs of successful tests")
for category in args.categories:
	passed_tests_names = per_category_passed_tests[category]
	# build path to test log: <base_directory> / <category> / output / <log file name>
	indir = os.path.join( base_directory, category, "output" )
	for test_name in passed_tests_names:
		# test log files are expecte to be named <log file name> = <test name>-output.log
		filename = test_name + "-output.log"
		input_file = os.path.join( indir, filename )
		if not os.path.exists( input_file ):
			# silently skip removing this log, the test may have produced none
			continue
		try:
			os.remove(input_file)
		except FileNotFoundError as e:
			print(f"{input_file} is not found:", e)
		except OSError as e:
			print(f"{input_file} is a directory:", e)
		except BaseException as e:
			print(f"{input_file}: unknown error:", e)

sys.exit( 0 if total_num_failures == 0 else 1 )
