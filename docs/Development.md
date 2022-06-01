
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

### General guidelines

* all functions should have `doxygen`-friendly documentation
* minimise the use of pre-processor macros (use C++11 `constexpr` instead)
* when closing a block (either `#endif` or `}`) and the block was long (whatever
long may be), add a comment on what it is that is being closed

### Naming
* classes: UpperCamelCase (e.g., MatrixContainer)
* class methods: lowerCamelCase (e.g., getElement)
* global functions: lower case, multi-word names separated by underscores
* variables: lower-case, underscore-separated
* template parameter type names: UpperCamelCase
* template parameter variables: lower-case, underscore-separated
* types (defined with typedef or using): lower case, separated by underscores,
terminating in `_type`


### Code formatting
* maximum line length is 200 characters, with the following exceptions
  * comment blocks are capped at 80 chars per line
  * argument lists (including template arguments) are capped at 80 characters per line


* indents should be *tabs*, not spaces
* alignment should be done using spaces, not tabs
* essentially any line that ends in `{`, `(`, or similar increases the
current number of indents by one and vice versa
* write `if( `, not `if (` (also for `for`, etc.)
* lines cannot have trailing whitespace
* files should end with an empty line


* be gratuitous with spaces and parenthesis: anything that could possibly be
construed as confusing or ambiguous should be clarified with spaces and
parentheses if that removes (some of the) possible confusion or ambiguity
* in particular, whenever it is legal to put one or more spaces, put one
(e.g., `if( x == 5 )` instead of `if( x==5 )`)
* in particular, only write `<<` or `>>` when doing bit shifts, not when
performing template magic
* exceptions from above rules:
  * negation of a condition (write `!condition` instead of `! condition`)
  * pointer or reference (write `&variable` instead of `& variable`)


* `#ifdef`, `#else`, `#endif` etc are never indented.
* OpenMP pragmas (or any pragma) are indented as regular code
* nested `ifdef`s etc. in close proximity of one another are indented by spaces


* no `if`, `for`, `while`, or any other control structure without curly
brackets, even if what follows is a single statement


* no lines with indents and curly brackets only: put curly brackets on the
same line as what starts that code block instead (only exception: code blocks
that are not started by standard C++ key words, but e.g. required pragmas
instead)


* include lines primarily ordered by
  1. standard includes
  2. external libraries
  3. internal headers/files


## Building and Testing infrastructure

To modify it, you should refer to the
[dedicated documentation](Build_and_test_infra.md).
