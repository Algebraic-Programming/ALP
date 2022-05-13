
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

# Development of ALP/GraphBLAS

This document introduces the reader to the development of ALP/GraphBLAS.

ALP/GraphBLAS is written in C++11 and is mainly composed of header files with
largely templated data structures and operations. This allows both

1. strict compile-time checking of the data types and of the algebraic
abstractions (typically encoded as template parameters: see the
[Semiring class](include/graphblas/semiring.hpp) for an example)
2. specialized code generation, increasing performance

## Code style tools and guidelines
ALP/GraphBLAS follows certain code style rules in order to ensure readability
and uniformity.

To apply these rules, the directory `tools` contains the script
`clang-format-linter.sh` to format (*lint*, in Unix jargon) the code
accordingly, based on the `clang-format` tool.
Version 11 or higher is requested for the settings to be applied; if you want to
use a different version, you can alias it in Bash before invoking
`tools/clang-format-linter.sh`, which directly calls the command
`clang-format-11`.
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

The style rules enforced by the tool are

- [x] lines are max 200 characters long, which means the line size is pretty
liberal to avoid weird re-flows
- [x] indents should be *tabs*, not spaces
- [x] alignment should be done using spaces, not tabs
- [x] essentially any line that ends in `{`, `(`, or whatever increases the
current number of indents by one and vice versa
- [x] argument lists (including template arguments) longer than 80 chars should
be broken over multiple lines
- [x] `if( `, not `if (` (also for `for`, etc.)
- [x] no lines with indents and curly brackets only: put curly brackets on the
same line as what starts that code block instead (only exception: code blocks
that are not started by standard C++ key words, but e.g. required pragmas
instead)
- [x] no lines ending with spaces
- [x] `#ifdef`, `#else`, `#endif` etc are never indented.
- [x] comment blocks are capped at 80 chars per line
- [x] include lines primarily ordered by
  1. standard includes
  2. external libraries
  3. internal headers/files

The following rules are also mandated, but cannot currently be applied via
`clang-format`; however, developers should abide by the following guidelines as
well:

* files should end with an empty line
* no `if`, `for`, `while`, or any other control structure without curly
brackets, even if what follows is a single statement
* OpenMP pragmas (or any pragma) are indented as regular code
* nested `ifdef`s etc. in close proximity of one another are indented by spaces

The following guidelines are not strictly requested nor enforced, but are
suggested to ensure readability and uniformity:

* be gratuitous with spaces and parenthesis: anything that could possibly be
construed as confusing or ambiguous should be clarified with spaces and
parentheses if that removes (some of the) possible confusion or ambiguity
* in particular, whenever it is legal to put one or more spaces, put one
(e.g., `if( x == 5 )` instead of `if( x==5 )`)
* in particular, only write `<<` or `>>` when doing bit shifts, not when
performing template magic
* when closing a block (either `#endif` or `}`) and the block was long (whatever
long may be), add a comment on what it is that is being closed
* all functions should have `doxygen`-friendly documentation
* minimise the use of pre-processor macros (use C++11 `constexpr` instead)

## Building and Testing infrastructure

To modify it, you should refer to the
[dedicated documentation](Build_and_test_infra.md).
