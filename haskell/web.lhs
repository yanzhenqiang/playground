= Client-side compiler =

++++++++++
<script>
function hideshow(s) {
  var x = document.getElementById(s);
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
</script>
++++++++++

Our next goal is a browser-based edition of our compiler.

== Crossly ==

We first add a few features related to WebAssembly. Firstly, the `wasm` command
compiles Haskell to C intended to be compiled to wasm. The resulting binary
assumes the host environment supplies a few functions such as `env.getchar`.
Secondly, the `warts` command prints the RTS generated by the compiler, also as
C code intended to be compiled to wasm. We later use this to build compilers
that can go directly from Haskell to wasm.

As usual, a bunch of other changes come along for the ride.
Recall we require a fixity declaration to precede the use of its corresponding
operator, which forces us to concatenate module sources in a particular order.
We remove this wart by adding a new phase. Once done, not only may we paste
together modules in any order, but we may also declare fixities anywhere within
a module.

During parsing, operators have the same precedence. When a chain of two or more
appear in a row, we abuse the syntax tree to store them in a right-associative
list, for example: `[1 + 2, * 3, - 4, + 5]`.

For patterns, we use the list field of a `PatCon` value; a made-up data
constructor `"{+"` indicates the beginning of such a list. Expressions are
clumsier; we bookend chains with the made-up basic combinators `"{+"` and
`"+}"`, and fashion a list out of `A` and `V` nodes.

By the time we call `patternCompile`, we have access to all modules. During
this phase, we traverse the syntax tree, and we re-associate each specially
marked infix chain now that we can look up the fixities of all operators.

The algorithm is conceptually straightforward. Starting from the first binary
infix expression, that is, two operands and one operator, for each operator and
operand we add on the right, we walk down the right spine of the current syntax
tree until we reach a node of higher precedence; leaf nodes are considered to
have maximum precedence. Then we insert the operator and operand at this point.
We also check for illegal infix operator conflicts.

The code is messy due to a couple of wrinkles. Firstly, we have two distinct ad
hoc representations of lists for holding infix chains. Secondly, we temporarily
mark operands with more ad hoc conventions to avoid descending too far when
reshaping syntax trees. For example, in the expression `1 + (2 + 3) * 4`, the
subexpression `(2 + 3)` is atomic.

We only allow top-level fixity declarations. We could add support for scoped
fixity declarations with yet more ad hoc encodings that we later use to create
scoped fixity lookup tables that override the global ones.

We do some housekeeping. Given a `Neat`, type inference had produced a tuple of
a particular type that contained the data needed by the next phase. We change
it to produce a new `Neat` with an updated `typedAsts` field, so there's one
fewer data type to occupy our thoughts and APIs. We no longer need to pick out
specific fields to pass to the next phase, as we simply pass everything.

We take a first stab at top-level type declarations. We treat them similarly to
default typeclass methods, in that during type inference, we trust the symbol
has its annotated type, and only afterwards that we verify the annotated type
matches the inferred type. It's more complex because we must process an entire
strongly connected component at a time.

Adding modules has made a mess of our various functions for looking up data
constructors, top-level variables, typeclasses, and so on. We reorganize them
a little to standardize the logic for searching through the list of imports.
This makes it easier to add support for lists of export symbols.

++++++++++
<p><a onclick='hideshow("Ast3");'>&#9654; Toggle `Ast3.hs`</a></p><div id='Ast3' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/Ast3.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

++++++++++
<p><a onclick='hideshow("Parser3");'>&#9654; Toggle `Parser3.hs`</a></p><div id='Parser3' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/Parser3.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

++++++++++
<p><a onclick='hideshow("Typer4");'>&#9654; Toggle `Typer4.hs`</a></p><div id='Typer4' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/Typer4.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

++++++++++
<p><a onclick='hideshow("party2");'>&#9654; Toggle `party2.hs`</a></p><div id='party2' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/party2.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

== Precisely ==

Before proceeding, we crudely implement arbitrary-precision integers to enable
some cool demos. It also exercises our code that handles typeclasses.

Unlike standard Haskell, we add a `Ring` typeclass rather than `Num`. The `(+)`
and `(*)` operators should be reserved for rings, and there are uses for rings
that are not also instances of the `Num` typeclass. For example, Gaussian
integers are great for indexing a 2D rectangular board, especially when we want
to rotate by a right angle (multiplication by 'i') or talk about the cardinal
directions (which correspond the units).

The integers are https://en.wikipedia.org/wiki/Initial_algebra[the initial
ring], so we treat the integer constant 'n' as `fromInteger n`; if this results
in ambiguity, then we drop `fromInteger`.

Thus the laws that we know to be true in our bones, such as `a*(b + c) = a*b +
a*c`, will never lead us astray. We must explicitly write `fromIntegral` to,
say, map a `Word32` to a `Word64`. Other languages convert silently, and wind
up defying our algebraic intuition.

To represent an integer, we use a list of `Word32` numbers, plus a boolean to
represent its sign. For GHC compatibility we call the function
`integerSignList` instead of pattern matching on `Integer` values.

We implement schoolbook algorithms for basic arithmetic, which is
straightforward except for division. I realized that when doing long division
by hand, to find the next digit of the divisor, I pick something that seems
reasonable via a method that seems partly subconscious! How can we possibly
code this?

Luckily, there is a simple algorithm that makes good guesses. See Knuth, 'The
Art of Computer Programming'.

We rename `div` and `mod` to `quot` and `rem`, then introduce wrappers for
`div` and `mod`. Now our divisions behave correctly, though it is sad that
`div` and `mod` need more instructions. (FORTRAN set an unfortunate precedent
of truncating division to zero, ultimately
https://github.com/WebAssembly/design/issues/250[forcing languages like C and
WebAssembly and even hardware to conform].)

Our treatment of integer literals causes a bootstrapping issue. Suppose a
literal "0" is to be converted to an `Int`. Then our compiler applies the `Int`
edition of `fromInteger` to the `Integer` 0, which involves a call to `mpView`,
whose implementation needs the `Int` 0. If we simply code this as `0`, then we
wind up with a circular definition, because our compiler would insert another
`fromInteger` call. We work around this with a definition that bypasses
overloading by returning `ord '\0'`.

++++++++++
<p><a onclick='hideshow("BasePrecisely");'>&#9654; Toggle `BasePrecisely.hs`</a></p><div id='BasePrecisely' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/BasePrecisely.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

== Wasm to Haskell? ==

It'd be nice to write wasm ourselves, but for expedience, we rely on Clang to
compile our runtime system to a wasm binary which we manipulate.
WebAssembly turns out to be pleasantly malleable. A binary breaks up into
independent sections, and we can add, delete, or modify sections before
stitching them together again.

We write a tool that just does enough wasm parsing to print the sections of a
wasm file we want in the form of a Haskell module, and run this on the output
of `crossly warts` to create `WartsBytes.hs`.

++++++++++
<p><a onclick='hideshow("warts2hs");'>&#9654; Toggle `warts2hs.hs`</a></p>
<div id='warts2hs' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/warts2hs.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

The RTS includes generate code that wraps foreign imports. It can only be used
with programs that declare the same foreign imports.

== Webby ==

We build a compiler that goes directly from Haskell to wasm. The code does
little more than dumping the runtime system wasm binary along with bytes
describing the initial contents of the heap.

For now we look for the `main` function in the `Main` module and export it as
the `go` function; export declarations are ignored.

++++++++++
<p><a onclick='hideshow("Webby");'>&#9654; Toggle `Webby.hs`</a></p>
<div id='Webby' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/Webby.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++

== webby.wasm ==

To build a browser-based compiler, we essentially run the previous compiler on
itself, thus producing a wasm binary that can translate Haskell directly into
wasm.

One change is needed: we swap `System1.hs` for `SystemWasm.hs`. The
Linux version declares foreign imports for those in our runtime system for
Linux. The wasm version declares foreign imports for functions provided by
the host environment.

++++++++++
<p><a onclick='hideshow("SystemWasm");'>&#9654; Toggle `SystemWasm.hs`</a></p>
<div id='SystemWasm' style='display:none'>
++++++++++

------------------------------------------------------------------------
include::inn/SystemWasm.hs[]
------------------------------------------------------------------------

++++++++++
</div>
++++++++++