
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


# Development of ALP

This document introduces the reader to the development of ALP.

ALP is written in C++11 and is mainly composed of header files with largely
templated data structures and operations. This allows both

1. strict compile-time checking of the data types and of the algebraic
abstractions (typically encoded as template parameters: see the
[Semiring class](include/graphblas/semiring.hpp) for an example)

2. specialized code generation, increasing performance


## Code style and guidelines

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
   guards, and includes before the first line of code).

As the saying goes, exceptions prove the rules. For example, rule #3 could be
viewed as a specific exception to rule #8. Exceptions that are not
self-contained in the above set include:

1. one long program line under rule #7 may be arbitrarily spread over two lines
   even if it runs counter rule #3-- but not if it would spread over more than
   two lines;

2. OpenMP pragmas ignore rule #6;

3. the 80-character limit is not strictly enforced. For example, an OpenMP macro
   of 83 characters on a single line is better readable than one split over two;

4. brackets in code bodies that limit the scope of some of the declaration
   within the body, may, contrary to rule #3, appear alone on a single line.


## Code style by examples:

- `if( `, not `if (`;

- lines should never end in tabs or spaces;

- `if( x == 5 )` instead of `if( x==5 )`;

- only write `<<` or `>>` when doing bit shifts, never for nested templates;

- the following is correct, not a single line, nor without curly brackets;

```
if( ... ) {
	return SUCCESS;
}
```

```
/*
 * copyright info
 */

- the following is correct w.r.t. vertical spacing;

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

// note two empty lines follow:


namespace alp {

	// ...

}

#endif

// note an empty line follows

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

To apply these rules, the directory `tools` contains the script
`clang-format-linter.sh` that formats (*lint*, in Unix jargon) the code
accordingly, based on the `clang-format` tool. Version 11 or higher is requested
for the settings to be applied; if you want to use a different version, you can
alias it in Bash before invoking `tools/clang-format-linter.sh`, which otherwise
directly calls the command `clang-format-11`.

This tools is available in the standard repositories of the main Linux
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


# Code quality

Some major rules on code quality includes:

1. files always display the copyright and license header, and documents the
   initial author information and date of file creation;

2. limit the use of macros and in particular, never leak macro definitions to
   user code;

3. do not use `using` in a way that leaks to user code;

4. separate includes by their source -- e.g., a group of STL includes followed
   by utility header includes;

5. code documentation uses doxygen format;

6. use `constexpr` fields or functions in favour of any pre-processor macros.


# Building and Testing infrastructure

To modify it, you should refer to the
[dedicated documentation](Build_and_test_infra.md).

