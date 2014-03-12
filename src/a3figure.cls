%%
%% This is file `a3figure.cls',
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{a3figure}
              [2005/11/28 v0.0
 style file for an a3 figure for dutmsc.cls]
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%    MSc Thesis Definitions
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand*{\mscDepartment}[1]%
            {\def\mscdepartment{#1}}
\newcommand*{\mscFaculty}[1]%
            {\def\mscfaculty{#1}}
\newcommand*{\mscName}[1]%
            {\def\mscname{#1}}
\newcommand*{\mscTitle}[1]%
            {\def\msctitle{#1}}
\newcommand*{\mscSubTitle}[1]%
            {\def\mscsubtitle{#1}}
\newcommand*{\mscDate}[1]%
            {\def\mscdate{#1}}
\newcommand*{\mscBackPicture}[1]%
            {\def\mscbackpicture{#1}}
\newcommand*{\mscReaderOne}[1]%
            {\def\mscreaderone{#1}}
\newcommand*{\mscReaderTwo}[1]%
            {\def\mscreadertwo{#1}}
\newcommand*{\mscReaderThree}[1]%
            {\def\mscreaderthree{#1}}
\newcommand*{\mscReaderFour}[1]%
            {\def\mscreaderfour{#1}}
\newcommand*{\mscKeyWords}[1]%
            {\def\msckeywords{#1}}
% \mscDepartment and \mscFaculty needed for title pages
% \mscDepartment{Design, Integration and Operations of Aircraft and Rotorcraft}
% \mscFaculty{Aerospace Engineering}
\mscName{}
\mscTitle{}
\mscSubTitle{}
\mscKeyWords{}
\mscDate{}
\mscBackPicture{}
\mscReaderOne{}
\mscReaderTwo{}
\mscReaderThree{}
\mscReaderFour{}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%    PAPER SIZE DEFINITIONS: cf STANDARD BOOK CLASS
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a3paper}
   {\setlength\paperheight {420mm}%
    \setlength\paperwidth  {297mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility
\@openrighttrue
\else
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{a4paper,11pt,twoside,onecolumn,final,openright}
\ProcessOptions
\input{bk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@arabic\c@chapter}
\renewcommand \thesection {\thechapter.\@arabic\c@section}                      % 
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}               % \renewcommand\thesubsection   {\thesection-\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}        % \renewcommand\thesubsubsection{\thesubsection-\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}          % \renewcommand\theparagraph    {\thesubsubsection-\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}           % \renewcommand\thesubparagraph {\theparagraph-\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{%
    \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{roman}}
\newcommand\mainmatter{%
    \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand\part{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont                                                        % \sffamily
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \partname\nobreakspace\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont                                                        % \sffamily
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
               \if@openright
                \null
                \thispagestyle{plain}%
                \newpage
               \fi
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\Large\bfseries}}                % \sffamily\Large\bfseries
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}              % \sffamily\Large\bfseries
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}         % \sffamily\Large\bfseries
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}          % \sffamily\Large\bfseries
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}        % \sffamily\Large\bfseries
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Commands
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Environments
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
% --- Define Annexdote Style ---
\RequirePackage{ntheorem}
\newcounter{Anecdote}
\setcounter{Anecdote}{1} \theoremstyle{plain}
%\theoremheaderfont{\slshape}
\theoremheaderfont{\scshape} \theorembodyfont{\upshape}
\theoremindent1.5cm
\newtheorem{anecdote}{Anecdote}[Anecdote]
%\renewcommand{\theanecdote}{\thechapter.\arabic{anecdote}}
\renewcommand{\theanecdote}{\arabic{anecdote}}
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  TitlePages
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\if@compatibility
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Part, Chapter, Section, Subsection, ...
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{2}    % set to 2 to reduce table of contents depth
\newcommand\tableofcontents{%
    \thispagestyle{empty}  % empty
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \contentsname}{\contentsname}}   %\MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    \cleardoublepage
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries \sffamily #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries \sffamily
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@section{\sffamily \@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\sffamily \@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\sffamily \@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\sffamily \@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\sffamily \@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename}%
      \@mkboth{\listfigurename}%          \@mkboth{\listfigurename}%
              {\listfigurename}%          {\listfigurename}
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@figure{\sffamily \@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename}%
      \@mkboth{%
          \listtablename}%
         {\listtablename}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
      \@mkboth{\bibname}{\bibname}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\indexname}%
                        {\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 1cm 
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Table of Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
% \if@twoside
% \else
%   \raggedbottom
% \fi
% \if@twocolumn
%   \twocolumn
%   \sloppy
%   \flushbottom
% \else
  \onecolumn
% \fi
%--------------------------------------------------------------------------------------
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  HyperReference
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% pstricks loads xcolor, which should be loaded before hyperref
\RequirePackage[hyperref]{xcolor} % load xcolor with hyperref support
\RequirePackage{pstricks,pst-node}
%%
%%
% \RequirePackage[dvips,ps2pdf,pdfauthor=\mscname]{hyperref}
\RequirePackage[dvips,pdfauthor=\mscname]{hyperref}
\RequirePackage{colortbl}
%
\definecolor{darkred}{rgb}{0.5,0,0}
\definecolor{darkgreen}{rgb}{0,0.3,0}
\definecolor{darkblue}{rgb}{0,0,0.5}
\definecolor{darkbrown}{rgb}{0.28,0.07,0.07}
%
\hypersetup{colorlinks=true,
            citecolor=darkblue,
            urlcolor=darkblue,
            linkcolor=darkblue,
            menucolor=darkblue,
            anchorcolor=red,
            pagecolor=cyan,
            pdfborder={0 0 0},
            bookmarksnumbered=true}  % adds chapter numbers to pdf bookmarks
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Extra Packages
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
% \RequirePackage{fncychap}
% \if@a4paper
% use geometry package instead of the depreciated a4wide package
% \RequirePackage[headheight=13.6pt,bindingoffset=1cm,left=1in,top=3.5cm,height=22.5cm,width=35cm,headsep=1.0cm,footskip=1.5cm]{geometry}
\RequirePackage[bindingoffset=1cm,left=1in,top=5.5cm,height=22.cm,width=35cm,headsep=1cm,footskip=1.5cm]{geometry}
% \fi
\RequirePackage{fancyhdr}
\RequirePackage[leftbars]{changebar}
\RequirePackage{theorem}
\RequirePackage{graphicx}
\RequirePackage{psfrag}
\RequirePackage{nomentbl}   % LOCAL use nomentbl instead of nomencl, it uses a longtable environment to typeset the symbols list, 
                            % so that you can add the unit of symbols in an extra column
\RequirePackage{makeidx}
\RequirePackage{ifthen}
\RequirePackage{eso-pic}
% \RequirePackage[]{./local/fncychap/fncychap}   % LOCAL - old version
% \RequirePackage{fncychap}                      % dont use fncychap, doesnt work properly
\RequirePackage{sidecap}
\RequirePackage{amsmath,amssymb,nicefrac,eurosym}
\RequirePackage{amsfonts}
% \RequirePackage[style=default,labelformat=parens,format=hang]{subfig}
\RequirePackage[labelfont={rm,md,up},format=hang]{subfig}
\RequirePackage{caption}
\captionsetup{format=hang} % name multiline captions indent just like the captions of the subfigures
\RequirePackage{verbatim}

%%
%% Reset rule widths and the header width
\renewcommand{\headwidth}{\textwidth}%
\renewcommand{\headrulewidth}{0.5pt}%
% \renewcommand{\footrulewidth}{0pt}%
% \addtolength{\headheight}{2.5pt}%
% \addtolength{\headsep}{20pt}%

%
% --- Use Acronyms ---
\RequirePackage{acronym}
\renewcommand{\ac}[1]{%
  \expandafter\ifx\csname ac@#1@\thechapter\endcsname\relax%
    \acf{#1}%
    \expandafter\gdef\csname ac@#1@\thechapter\endcsname{x}%
  \else%
    \acs{#1}%
  \fi}
%
%
%--------------------------------------------------------------------------------------
\renewcommand{\captionfont}{\small\sffamily}
\renewcommand{\captionlabelfont}{\small\bfseries\sffamily}
\setlength{\captionmargin}{20pt}



%% ========================================================================
%% Headers and footers
%% ========================================================================
%% No header or footer on the title page
% \AtBeginDocument{\thispagestyle{plain}}
%% Use funky headers and footers
% \pagestyle{fancy}

%% Normal headers and footers 
% \renewcommand{\chaptermark}[1]{\markboth{#1}{}}
% \renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
% \fancyhead[LO,RE]{\bfseries\thepage}%
% \fancyhead[LE]{\bfseries\leftmark}%
% \fancyhead[RO]{\bfseries\rightmark}%
% clear all footer fields
% \fancyfoot{}%
% %% Lower case header content
% \renewcommand{\chaptermark}[1]{%
%   \markboth{#1}{}%
% }
% 
%REQUIRES "FANCYHDR.STY"
%-------------------------------------------------------------------------------------------
%                            Header definitions... (Clark Borst)
%-------------------------------------------------------------------------------------------
\pagestyle{fancyplain}
\addtolength{\headheight}{0.2\baselineskip} %add some space to headheight to eliminate LaTeX "Overfull \vbox..." warnings
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries\rightmark}}
\rhead[\fancyplain{}{\bfseries\leftmark}]{\fancyplain{}{\bfseries\thepage}}
\cfoot{}


% Headers and footers for plain (blank) pages
\fancypagestyle{plain}{%
  \fancyhf{}%
  %\fancyhead[RO,LE]{\bfseries \thepage}%
  \fancyfoot[C]{\bfseries \thepage}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TOC / LOF / LOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \newcommand{\lofname}{List of Figures}
% \newcommand{\lotname}{List of Tables}

\newcommand{\toc}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
}

\newcommand{\toclof}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
\addcontentsline{toc}{chapter}{\numberline{}\listfigurename}  % comment line to prevent "List of Figures" entry in Table of Contents
    \listoffigures\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
}

\newcommand{\tocloflot}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
\addcontentsline{toc}{chapter}{\numberline{}\listfigurename}  % comment line to prevent "List of Figures" entry in Table of Contents
    \listoffigures\cleardoublepage
\addcontentsline{toc}{chapter}{\numberline{}\listtablename}  % comment line to prevent "List of Tables" entry in Table of Contents
    \listoftables\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
    \cleardoublepage
}

% REVIEWED: MARTIJN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chapter with no numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\nonumchap}[1]{%
    \thispagestyle{empty}
    \phantomsection % Added by Martijn van de Giessen
    \addcontentsline{toc}{chapter}{\numberline{}#1}
    \chapter*{#1}
    \chaptermark{#1}
    \markboth{#1}{#1}
}

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Print nomenclature list
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printnomencl}{
\phantomsection % Added by Martijn van de Giessen
\addcontentsline{toc}{chapter}{\numberline{}\nomname}
\markboth{\nomname}{\nomname}
\printnomenclature[0.8in]
\cleardoublepage
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print bib list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printbib}[1]{
\setlength{\bibitemsep}{\baselineskip}
\cleardoublepage
\phantomsection % Added by Martijn van de Giessen
\addcontentsline{toc}{chapter}{\numberline{}\bibname}
% \lhead{\nouppercase{\rightmark}}
% \rhead{\nouppercase{\leftmark}}
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries \bibname}}
\rhead[\fancyplain{}{\bfseries \bibname}]{\fancyplain{}{\bfseries\thepage}}
\bibliography{#1}
\cleardoublepage
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries\rightmark}}
\rhead[\fancyplain{}{\bfseries\leftmark}]{\fancyplain{}{\bfseries\thepage}}
\cfoot{}
}



%------------------------------------------------------------------------------------%
%                            EXAMPLE ENVIRONMENT                                     %
%------------------------------------------------------------------------------------%
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}

\renewcommand{\Re}{\mathfrak{Re}}
\renewcommand{\Im}{\mathfrak{Im}}

\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}


%
% List of symbols with units
%
\renewcommand{\nomname}{List of Symbols}
\setlength{\nomitemsep}{-0.5\parsep}
% latin symbol
\newcommand{\lsymb}[5]
    {#1\nomenclature[A#3]{#1}{#2}{#4}{#5}}
% greek symbol
\newcommand{\gsymb}[5]
    {#1\nomenclature[G#3]{#1}{#2}{#4}{#5}}
% subscripts
\newcommand{\subscr}[5]
    {\nomenclature[Z]{$_{#1}$}{#2}{#4}{#5}}
% superscripts
\newcommand{\superscr}[5]
   {\nomenclature[X]{$^{#1}$}{#2}{#4}{#5}}

\makeatletter
\newcommand\BackgroundPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vfill
    \centering\includegraphics{#1}
    \vfill
}}}
\newcommand\TuPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vfill
    \centering\includegraphics{#1}
}}}
\makeatother
%--------------------------------------------------------------------------------------
\newcommand{\eqnref}[1]{Eq.\ (\ref{#1})}
\newcommand{\figref}[1]{Figure~\ref{#1}}
\newcommand{\tabref}[1]{Table~\ref{#1}}
%--------------------------------------------------------------------------------------

\renewcommand{\maketitle}{%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main title with picture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{titlepage}
\ifthenelse{ \equal{\mscbackpicture}{} }{}{
\AddToShipoutPicture*{\BackgroundPicture{\mscbackpicture}}
}
\AddToShipoutPicture*{\TuPicture{./figs/TUD_bies_LR_A4_ENG}}
 \begin{center}
 \large{\textsc{Master of Science Thesis}}%\\
 \end{center}%
 \newcommand{\HRule}{\rule{\linewidth}{1mm}}
 \setlength{\parindent}{0mm}
 \setlength{\parskip}{0mm}
 \vspace*{\stretch{2}}
 \HRule
 \begin{flushright}
 \huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[5mm]
%  \Huge{\textbf{\textsf{\msctitle}}}
% 
%  \Large{\textbf{\textsf{\mscsubtitle}}}
% 
%  \\[5mm]
 \Large{\textbf{\textsf{ \mscname}}}
 \end{flushright}
 \HRule
 \\ \mscdate \\
% 
%  \mscdate
% 
 \vspace*{\stretch{2}}
 %\begin{center} \epsfig{file=cslogo.eps,width=3cm} \end{center}
 %
%  \begin{center}
%  \begin{tabular}{lcr}
%  Faculty of \mscfaculty & $\cdot$ & Delft University of Technology\\
%  \end{tabular}
%  \end{center}
   \hspace{1cm}
\end{titlepage}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Second title page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\clearemptydoublepage
\thispagestyle{empty}
\setcounter{page}{1}
\begin{center}
\Huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[1cm]
\Large{\textsc{Master of Science Thesis}}\\[2cm]
% \Large{\textsc{Master of Science Thesis}}\linebreak\vspace{2cm}
\Large{For obtaining the degree of Master of Science in Aerospace Engineering at Delft University of Technology}\\[5cm]
% \Large{For obtaining the degree of Master of Science in Aerospace Engineering at Delft University of Technology}\linebreak\vspace*{5cm}
\Large{\mscname}\\[1cm]
% \Large{\mscname}\linebreak\vspace*{1cm}
\Large{\mscdate}
\end{center}
\vspace*{\stretch{2}}
%\begin{center} \epsfig{file=cslogo.eps,width=3cm} \end{center}
%
\begin{center}
\begin{tabular}{lcr}
Faculty of \mscfaculty & $\cdot$ & Delft University of Technology
% Faculty of \mscfaculty & $\cdot$ & Delft University of Technology\\
\end{tabular}
\end{center}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\vspace*{15cm}
\vspace*{\stretch{3}}
\thispagestyle{empty}
\begin{figure}[h!]
  \includegraphics{./figs/tud}
  \label{fig:TUDlogo}
\end{figure}
Copyright \copyright\ \mscname \\
% \newline\noindent
All rights reserved.
\cleardoublepage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Readers Page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\thispagestyle{empty}

\begin{center}
\begin{tabular}{c}
%      \large \textsc{Delft University Of Technology}\\
     \textsc{\large Delft University Of Technology}\\
     \textsc{Department Of} \\
     \textsc{\mscdepartment}
\end{tabular}
\end{center}

\vspace{1cm}
%
%\large

The undersigned hereby certify that they have read and recommend
to the Faculty of \mscfaculty\ for acceptance a thesis
entitled \textbf{``\msctitle''} by
\textbf{\mscname} in partial fulfillment of the
requirements for the degree of \textbf{Master of Science}.

\vspace{4cm}

\mbox{} \hfill Dated: \underline{ \mscdate }

\vspace{1cm}

\noindent
Head of department:     \hfill \underline{ \hspace{7cm} }\\
\mbox{}                 \hfill  \mscreaderone\\
\\
\\
Supervisor:             \hfill \underline{ \hspace{7cm} }\\
\mbox{}                 \hfill  \mscreadertwo\\
\ifthenelse{ \equal{\mscreaderthree}{} }{}{\\
\\
Reader:                 \hfill \underline{ \hspace{7cm} }\\
\mbox{}                 \hfill \mscreaderthree\\
}
\ifthenelse{ \equal{\mscreaderfour}{} }{}{\\
% \\
\mbox{}                 \hfill \underline{ \hspace{7cm} }\\
\mbox{}                 \hfill \mscreaderfour\\
}
%
\newpage\clearemptydoublepage}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill out PDF author - title
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\setThesisInfo}{
\makeglossary
\makeindex
\makenomenclature
\hypersetup{pdftitle    = Masters Thesis: \msctitle,
            pdfauthor   = \mscname,
            pdfsubject  = \mscsubtitle,
            pdfkeywords = \msckeywords}
}


%--------------------------------------------------------------------------------------
% \lfoot[\scriptsize \textsf{\mscname}]{\scriptsize \textsf{M.Sc. thesis}}
% \rfoot[\scriptsize \textsf{M.Sc. thesis}]{\scriptsize \textsf{\mscname}}
%--------------------------------------------------------------------------------------
\sloppy
\endinput
%%
%% End of file `dutmsc.cls'.
