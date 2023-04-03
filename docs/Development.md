
<pre>
  Copyright 2021 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
</pre>


# ALP Development Style Guide

This document introduces the reader to the development style of ALP.

ALP is written in C++11 and is mainly composed of header files with largely
templated data structures and operations. This allows both

1. strict compile-time checking of the data types and of the algebraic
abstractions (typically encoded as template parameters: see the
[Semiring class](include/graphblas/semiring.hpp) for an example);

2. specialised code generation, increasing performance.

Common patterns include SFINAE and in particular its combination with
(algebraic) type traits, as well as copious use of `static_assert` and
`constexpr`. The choice of ANSI C++11 is to balance the benefits of these more
modern C++ constructs with the typical reluctance of applying the latest and
greatest in software development tooling within production codes.

Given that this is a template library, there are both rigid code styles as well
as more rigid coding patterns to ensure the overall quality of the template
library-- these are detailed in their respecive sections. This document also
includes a brief description of code style tools included with the repository,
as well as a section on the use of the available build and test infrastructure.

First, however, this section concludes with some brief comments on the overall
code structure.

## Encapsulation

Template code that should not be exposed to ALP programmers (i.e., users of the
ALP programming interface) should be encapsulated in an internal namespace such
as, e.g., `grb::internal`. Non-templated code that should not be exposed to ALP
programmers should be defined within `.cpp` files. Only functionality that is
called by templated code should be exported during compilation of the ALP
libraries that ALP programmers would link against. All code that may be used by
ALP programmers should be documented thoroughly.

## Utilities

Utility functions that could be useful by ALP programmers and not just by ALP
developers, should unambiguously be housed in the `./include/graphblas/utils`
directory, with the interfaces made availble through the corresponding
`grb::utils` namespace. These functionalities should therefore and ideally *not*
be included in an internal namespace.

## Test utilities

Utility functions that are *only* useful for ALP unit, smoke, and/or performance
tests should unambiguously be housed in the `./tests/utils` directory. It should
never be included with code functionalities for ALP programmers. These
functionalities should never be included with the template library, neither as a
header that could be invoked by ALP programmers, nor within an internal
namespace or within an internal `.cpp` file.


# Code style guidelines

ALP follows certain code style rules in order to ensure readability and
uniformity. An informal summary of the main points follow:

1. alignment uses spaces while indentation uses tabs;

2. indentation is increased after a line break that does not end with `;`,
   increased after a line break with an unterminated `<`, `(` or `{` and
   decreased after matching `;`, `>`, `)`, and `}`. Opening and closing
   delimiters are the last, resp., first characters on every line-- i.e., the
   commonly accepted indentation pattern;

3. none of `;`, `<`, `(`, `{` should appear alone on a single line-- while if
   the opening delimiters like `<` follows a keyword it should do so
   immediately, without intermediate spaces;

4. when a closing delimiter is far (in a vertical space sense) from its opening
   pair, it should be followed by a comment that documents what it closes;

5. keywords that induce indentation include `private:`, `protected:`, and
   `public:`, which furthermore do not induce intermediate spaces between the
   keyword and the `:`;

6. indentation of pre-processor code (macros) uses spaces, not tabs, and ignores
   tab-based identation;

7. a single line has maximum length of about 80 characters, not including
   indentation, and never ends with white spaces (space characters or tab
   characters);

8. use spaces and parentheses liberally for increasing code readability and to
   limit ambiguity, including for if-else blocks or for-loop blocks that consist
   only of one (or an otherwise limited number of lines);

9. files always end with an empty line, and includes two empty lines before
   implementation starts (i.e., two empty lines after any comments, macro
   guards, and includes before the first line of code);

10. Classes and types use the CamelCase naming format, variables of any kind
    (static, constexpr, global, or members) use camelCase, while constants of
    any kind (static const, global const, constexpr const, etc.) use CAMELCASE.
    Names shall furthermore be both self-descriptive and short. Namespaces are
    camelcase.

As the saying goes, exceptions prove the rules. For example, rule #3 could be
viewed as a specific exception to rule #8. Exceptions that are not
self-contained in the above set include:

1. one long program line under rule #7 may be arbitrarily spread over two lines
   even if it runs counter rule #3-- but not if it would spread over more than
   two lines;

2. OpenMP pragmas may ignore rule #6-- they may follow regular tab-based
   indentation instead;

3. the 80-character limit is not strictly enforced. For example, an OpenMP macro
   of 83 characters on a single line is better readable than when split over
   two;

4. brackets in code bodies that limit the scope of some of the declaration
   within the body, may, contrary to rule #3, appear alone on a single line.


## Code style by examples:

- `if( ... ) {`, not `if (...) {` or any other variant;

- lines should never end with white space (tab or space characters);

- `if( x == 5 ) {` instead of `if( x==5 ) {`;

- only write `<<` or `>>` when doing bit shifts, never for nested templates;

- the following is correct. It would *not* be correct to put the whole block on
  a single line, nor would it be correct to write it without any curly brackets;

```
if( ... ) {
	return SUCCESS;
}
```

- the following is correct w.r.t. vertical spacing;

```
/*
 * copyright info
 */

/**
 * @file
 *
 * File documentation
 *
 * @author Author information
 * @date   Date of initial creation
 */

#ifndef MACRO_GUARD
#define MACRO_GUARD

// note that two empty lines follow:


namespace alp {

	// ...

}

#endif

// note that one empty line follows:

```

- encapsulation using curly bracket delimitors that both appear on a single
  line:

```
void f( ... ) {
	// some code block dubbed "A"
	// ...
	// end code block A
	size_t ret;
	{
		// some code block with ields and containers that are used *solely* for
		// for computing ret
		// ...
		ret = ...;
	}
	// some code that uses ret as well as fields, containers, and anything else
	// that was defined in code block A
}
```


# Code style tools

There currently exist two tools to help check developer's code styles: the Clang
linter script `clang-format-linter.sh`, and the `detectSuspiciousSpacing.sh`
script.

## Clang linter

To automatically and approximately correctly check whether code style rules are
followed properly, the directory `tools` contains the script
`clang-format-linter.sh` that formats (*lints*, in Unix jargon) the source code,
based on the `clang-format` tool.

Version 11 or higher of the tool is required. If you want to use a different
version, you can alias it in Bash before invoking
`tools/clang-format-linter.sh`, which otherwise directly calls the command
`clang-format-11`.

This tools is available in the standard repositories of all main Linux
distributions: for example, in Ubuntu you can install it with
`apt-get install clang-format-11`.

To list the script parameters, simply type

```bash
tools/clang-format-linter.sh -h
```

For example, to lint the file `tests/add15d.cpp` and see the lint'ed code on the
standard output, type

```bash
tools/clang-format-linter.sh tests/add15d.cpp
```

while to change the file in-place, add the `-i` option

```bash
tools/clang-format-linter.sh -i tests/add15d.cpp
```

Instead, to lint the whole ALP/GraphBLAS code-base in-place, type

```bash
tools/clang-format-linter.sh -i --lint-whole-grb
```

### Warning

This tool is only approximately correct in terms of the code style described
above(!)


## Automated detection of suspicious spacing

Many code reviews have exposed erroneous use of spaces, primarily due to editors
attempting to be helpful in automatically replicating code styles like
indentations. Before committing code, a careful submitter may opt to execute
something like the following:

```
# go into a source directory where you have committed changes
$ cd include/graphblas/nonblocking
# **from within that directory** execute the helper script:
$ ../../../tools/detectSuspiciousSpacing.sh
```

If all is OK, the output of the above would print the following to the standard
output stream (which also immediately documents which patterns the script is
tailored to detect):

```
Detecting suspicious spacing errors in the current directory, /path/to/source/include/graphblas/nonblocking
	 spaces, followed by end-of-line...
	 tabs, followed by end-of-line...
	 spaces followed by a tab...
$
```

Seeing no `grep` output between the noted patterns (or between the last noted
pattern and the prompt) means that no such patterns have been found within any
source file in the current directory, including source files in a subdirectory
to the current path.


# Coding patterns for general code quality

Some major coding rules for maintaining high code quality include:

1. files always display the copyright and license header, and documents the
   initial author information and date of file creation;

2. limit the use of macros and in particular, never leak macro definitions to
   user code;

3. do not use `using` in a way that leaks to user code;

4. separate includes by their source -- e.g., a group of STL includes followed
   by a group of internal utility header includes, and so on;

5. code documentation uses doxygen format, and in particular the Javadoc style;

6. use `constexpr` fields or functions in favour of any pre-processor macros,
   and avoid global constants, especially those that leak to user code;

7. performance parameters are never hardcoded but instead embedded (and
   documented!) into the applicable `config.hpp` file.


# Building and Testing infrastructure

To use the build and test infrastructure, see the [main README](../README.md).
To modify it, you should refer to the
[dedicated documentation](Build_and_test_infra.md).


## Testing before committing

A careful committer may wish to run smoke or unit tests before committing to the
main repository. Such developers may wish to take note of the script contained
in the tests directory, `tests/summarise.sh`, which may be used to quickly
analyse a test log file: it summarises how many tests have passed, how many have
been skipped, and how many have failed.

Additionally, if at least one test has failed, or if none of the tests have
succeeded (indicating perhaps a build error), then the entire log will be
`cat`-ted.

A common use is to, in one terminal, execute:

```
$ cd build
$ make -j88 smoketests &> smoketests.log
```

While in another, and while the above command is running, to execute:

```
$ cd build
$ watch ../tests/summarise.sh smoketests.log
```

The second terminal then gives ``live'' feedback on the progress of the tests.

## Continuous integration

GitHub actions have been deployed to run smoke tests using both performance and
debug flags. These tests are run on standard images that do not include the
the datasets that some smoke tests require -- those tests are hence skipped.

An internal CI to the Computing Systems Lab at the Huawei Zurich Research Center
exists, but can only be triggered by its employees. This CI also performs unit
tests, in addition to smoke tests. At present, however, it also does *not*
employ images that have the required data sets embedded or accessible.

The `develop` and `master` branches are tested by the internal CI on a regular
schedule, in addition to being triggered on every push, and run a more
comprehensive combination of test suites and compilation (debug/release) flags.
Also release candidate branches (i.e., branches with names that match the
wild-card expression `*-rc*`) are subject to the same more extensive test suite.

All CI tests at present skip tests that require data sets, and therefore
developers are suggested to not skip running local tests manually, at least once
before flagging a merge request as ready and requesting a review. Even if at
some point the CI does provide datasets, the practice of developers
self-checking MRs is recommended as it naturally also induces greater robustness
across compilers and distributions.

