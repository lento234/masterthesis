This is a LaTeX thesis style for use at the Delft University of Technology. It
is based on the thesis style that has been used for a number of years at the
Chair of Control and Simulation of the Faculty of Aerospace Engineering.

This style is maintained by Wim Van Hoydonck 
(w.r.m.vanhoydonck@tudelft.nl).

Questions, praises, bug reports, flowers and chocolates can be send to the maintainer.


Visually, not a lot of changes have been made. The same visual style is used
for chapters and headings, the only difference being that the footers are
empty.

Improvements:
v1.0.8:
  - direct pdf output is now supported, so the three step process: latex -> dvi,
    dvi -> ps, ps -> pdf can be replaced by latex -> pdf.
    This has consequences for the figure formats that are supported. For the direct
    conversion from latex to pdf, about everything is supported except for eps files 
    (but just stick to pdf figures to be on the safe side).
  - The introductory chapter in the example thesis is in need of some serious attention.
v1.0.7:
  - added official logos and styles of the university (the old ones can still be
    used if you wish, see logos_tu.README), some simplification in the placement of 
    the bies on the front page.
  - small typos in the introductory chapter removed.
  - Next to the dos batch file (sort_symb.bat), a linux shell file (sort_symb.sh) 
    was added to sort the symbols list. 
    The latter can be executed from a shell using either of the following methods:
      $ source sort_symb.sh
    or:
      $ chmod u+x sort_symb.sh
      $ ./sort_symb.sh
    The dos batch file can be executed by clicking on it (I think).
v1.0.6:
  - The bibliography style now used is 'plain'. Packages natbib and apacite are not 
    used anymore. For this to work, a single line in the definition of printbib in 
    the style file had to be commented.
  - sort_symb.bat batch file included to sort the symbols automatically.
v1.0.5:
  - name of reference chapter was written in french on students computers, fixed.
  - fixed a bug reported by Harm Kuipers. the sorting of the greek, latin and 
    other symbols went wrong if the symbols did not appear in the text.
v1.0.4:
  - added the preserveurlmacro option to the breakurl package. Seems to work 
    better...
v1.0.3:
  - Sorting symbols did not work properly: $\dot{m}$ would appear before $m$
    in the list of symbols. This has been corrected by adding an extra argument
    to the commands \gsymb, \lsymb and \osymb.
    Thanks to Ingmar for reporting this.
  - I tried to make it more clear that "makeindex" is not a latex command, but a 
    separate program on the user's computer.
  - Chapters/sections without numbers (such as Table of contents, List of Figures,
    Bibliography, ...) are not indented any more.
v1.0.2:
  - With the hyperref package, captions were not split into multiple lines,
    now they are. This is achieved by adding the option breaklinks=true, to 
    the list of hypersetup options.
    Thanks to Nico for reporting this issue.
v1.0.1:
  - Ack. There was something wrong with the \loflotintoc command.
    The \listoffigures should directly be followed by \addcontentsline, after 
    which the \cleardoublepage comes and not the other way around.
v1.0:
  - Readers page can contain up to six readers and it is not necessary to edit
    the style file to change the number of readers, just (un)comment as
    required.  
  - Obsolete a4wide replaced with the geometry package with as result that
    textareas of the pages on the same sheet coincide.
  - Nomenclature changes: separate acronyms page merged in nomenclature list,
    with as result that they too are sorted with Makeindex.
    All symbol commands contain an optional argument that determines wether or
    not a symbol appears in the text where the command was issued.
    The following categories of symbols are available:
      - latin symbols     \gsymb
      - greek symbols     \lsymb
      - subscripts        \subscr
      - superscripts      \superscr
      - other symbols     \osymb
      - acronyms          \acron
  - Option added to include the list of figures and list of tables in the
    table of contents (\loflotintoc). If its argument is 1, the figure and
    table lists appear in the table of contents (with proper working
    references in the pdf output).
  - Hypersetup moved in the main thesis file just before the beginning of 
    the document body to ensure that the pdfauthor, pdftitle and 
    pdfkeywords are correct in the pdf output.


It does not ship with a lot of local packages (for Windows users, all 
required packages should be available when using Miktex).

List of local packages:
- apacite       (citations)
- breakurl      (breaking urls in sensible places)
- fncychap      (fancy chapter style)
- nomencl       (nomenclature list)

These packages should also be available with a recent version of Miktex, Tetex 
or TexLive.

