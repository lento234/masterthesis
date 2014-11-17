%%
%% This is file `dutmsc.cls',
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{dutmsc}
              [2012/10/07 v1.0.8
 Delft University of Technology Master's Thesis Style]

\LoadClass[a4paper,11pt,twoside,onecolumn,openright]{book}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%    Changes to the default book class style
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% make 'table of contents', 'list of figures' and 'list of tables' 
%% lowercase in the header
%%
\renewcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }%
\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename}%
      \@mkboth{\listfigurename}%
              {\listfigurename}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }%
\renewcommand\listoftables{%
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
    }%
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
\newcommand*{\mscReaderFive}[1]%
            {\def\mscreaderfive{#1}}
\newcommand*{\mscReaderSix}[1]%
            {\def\mscreadersix{#1}}
\newcommand*{\mscKeyWords}[1]%
            {\def\msckeywords{#1}}
%
% put list of figures and list of tables in table of contents
% if the argument is zero, they are not included, else, a line is 
% added to the toc-file.
\newcommand*{\loflotintoc}[1]%
            {\def\loflot_in_toc{#1}}
%
% \mscDepartment and \mscFaculty needed for title pages
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
\mscReaderFive{}
\mscReaderSix{}
\loflotintoc{}
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Fancy Chapter Headings
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% (must be loaded before hyperref)
\RequirePackage{./local/fncychap/fncychap}
%
% customizations
\makeatletter
  \ChNameVar{\raggedleft\Huge\rm}
  \ChNumVar{\Huge\rm}
  \ChTitleVar{\raggedleft\Huge\sf\bfseries}
  \ChRuleWidth{2pt}

  \renewcommand{\DOCH}{%
    \vskip -0.5\baselineskip
    \mghrulefill{\RW}\par\nobreak
    \CNV\FmN{\@chapapp}\space \CNoV\thechapter
    \par\nobreak
    \vskip -0.5\baselineskip
   }
  \renewcommand{\DOTI}[1]{%
    \mghrulefill{\RW}\par\nobreak
    \CTV\FmTi{#1}\par\nobreak
    \vskip 60\p@
    }
  \renewcommand{\DOTIS}[1]{%
    \mghrulefill{\RW}\par\nobreak
    \CTV\FmTi{#1}\par\nobreak
    \vskip 60\p@
    }
\makeatother
%
%
\RequirePackage{ifthen}%
\RequirePackage{ifpdf}%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  HyperReference
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% pstricks loads xcolor, which should be loaded before hyperref
\RequirePackage[hyperref]{xcolor} % load xcolor with hyperref support
%
% when pdf output is generated, do not use pstricks and the dvips driver for the hyperref package
\ifpdf
  \RequirePackage{hyperref}
\else
  \RequirePackage{pstricks,pst-node}
  \RequirePackage[dvips]{hyperref}
\fi
%%
%%
\RequirePackage{colortbl}
%
\definecolor{darkred}{rgb}{0.5,0,0}
\definecolor{darkgreen}{rgb}{0,0.3,0}
\definecolor{darkblue}{rgb}{0,0,0.5}
\definecolor{darkbrown}{rgb}{0.28,0.07,0.07}
\definecolor{black}{rgb}{0,0,0}
\definecolor{plotGreen}{rgb}{0,0.5,0}
\definecolor{plotRed}{rgb}{1.0,0,0}
\definecolor{plotBlue}{rgb}{0,0,1.0}
\definecolor{plotCyan}{rgb}{0,1.0,1.0}
\definecolor{plotGray}{gray}{0.25}
%
% \hypersetup{colorlinks=true,
%             citecolor=darkblue,
%             urlcolor=darkred,
%             linkcolor=darkblue,
%             menucolor=darkblue,
%             anchorcolor=red,
%             pagecolor=cyan,
%             breaklinks=true,
%             pdfborder={0 0 0},
%             bookmarksnumbered=true     % adds chapter numbers to pdf bookmarks
% }
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Extra Packages and Options
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%% use geometry package instead of the depreciated a4wide package
% \RequirePackage[headheight=14.6pt,bindingoffset=0cm,left=1in,top=3.5cm,height=22.5cm,width=14.9cm,headsep=1.0cm,footskip=1.5cm]{geometry}
% equal left and right margins
\RequirePackage[headheight=14.6pt,hmarginratio=1:1,top=3.5cm,height=22.5cm,width=14.9cm,headsep=1.0cm,footskip=1.5cm]{geometry}
\RequirePackage{fancyhdr}
%
\RequirePackage{graphicx}
%
% for pictures on title pages
\RequirePackage{eso-pic}
%
% postscript fragment replacements
\RequirePackage{psfrag}
%
% version 4.2 of nomencl
\RequirePackage[notintoc]{nomencl}[2005/09/22]
%
%\RequirePackage[labelfont={rm,md,up},format=hang]{subfig}
\RequirePackage{caption}
% multiline captions indent just like the captions of the subfigures
%\captionsetup{format=hang}
\RequirePackage{subcaption}
%
%
% 
% separate paragraphs instead of indenting them
\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}
%
% small sans-serif font family for captions
\renewcommand{\captionfont}{\small\sffamily}
\renewcommand{\captionlabelfont}{\small\bfseries\sffamily}
\setlength{\captionmargin}{20pt}
%
% better vertical spacing in tables
\newcommand\T{\rule{0pt}{2.6ex}}
\newcommand\B{\rule[-1.2ex]{0pt}{0pt}}
%
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}
%
% signature for the acknowledgements
\newcommand{\signature}{%
    \vspace*{15mm}
    \noindent
    Delft, The Netherlands \hfill \mscname\\
    \mscdate}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Nomenclature definitions
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% \renewcommand{\nomname}{List of Symbols}%
\renewcommand{\nomname}{Nomenclature}%
% vertical spacing in the nomenclature subsection titles
\newcommand{\rb}{15pt}%
\newcommand{\nomAname}{\raisebox{\rb}{Latin Symbols}}%
\newcommand{\nomGname}{\raisebox{\rb}{Greek Symbols}}%
\newcommand{\nomWname}{\raisebox{\rb}{Subscripts}}%
\newcommand{\nomXname}{\raisebox{\rb}{Superscripts}}%
\newcommand{\nomYname}{\raisebox{\rb}{Abbreviations}}%
\newcommand{\nomZname}{\raisebox{\rb}{Other Symbols}}%
%
\markboth{\nomname}{\nomname}
%
\renewcommand\nomgroup[1]{\vspace{10mm}%
    \ifthenelse{\equal{#1}{A}}{%
      \item[\LARGE\textbf{\textsf{\nomAname}}]}{%
      \ifthenelse{\equal{#1}{G}}{%
        \item[\LARGE\textbf{\textsf{\nomGname}}]}{%
        \ifthenelse{\equal{#1}{W}}{%
          \item[\LARGE\textbf{\textsf{\nomWname}}]}{%
          \ifthenelse{\equal{#1}{X}}{%
            \item[\LARGE\textbf{\textsf{\nomXname}}]}{%
            \ifthenelse{\equal{#1}{Y}}{%
              \item[\LARGE\textbf{\textsf{\nomYname}}]}{%
              \ifthenelse{\equal{#1}{Z}}{%
                \item[\LARGE\textbf{\textsf{\nomZname}}]}{%
            {}}}}}}}}
%
\setlength{\nomitemsep}{-0.3\parsep}
%
% for right-aligned units
\newcommand{\nomunit}[1]{%
  \renewcommand{\nomentryend}{\hspace*{\fill}#1}}
%
% latin symbol
\newcommand{\lsymb}[5][t]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[A#5]{#2}{#3\nomunit{#4}}}%
        {\nomenclature[A#5]{#2}{#3\nomunit{#4}}}}%
% greek symbol
\newcommand{\gsymb}[5][t]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[G#5]{#2}{#3\nomunit{#4}}}%
        {\nomenclature[G#5]{#2}{#3\nomunit{#4}}}}%
% other symbols
\newcommand{\osymb}[4][f]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[Z#4]{#2}{#3}}%
        {\nomenclature[Z#4]{#2}{#3}}}%
% subscripts
\newcommand{\subscr}[3][f]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[W]{$_{#2}$}{#3}}%
        {\nomenclature[W]{$_{#2}$}{#3}}}%
% superscripts
\newcommand{\superscr}[3][f]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[X]{$^{#2}$}{#3}}%
        {\nomenclature[X]{$^{#2}$}{#3}}}%
% acronyms
\newcommand{\acron}[3][f]
    {\ifthenelse{ \equal{#1}{t} }%
        {#2\nomenclature[Y]{\textbf{#2}}{#3}}%
        {\nomenclature[Y]{\textbf{#2}}{#3}}}%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  Header Definition
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
\pagestyle{fancyplain}
% \addtolength{\headheight}{0.2\baselineskip}   % add some space to headheight to eliminate LaTeX "Overfull \vbox..." warnings
% \setlength{\headheight}{13.6pt}                 % add some space to headheight to eliminate LaTeX warnigs of \headheight too small....
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\bfseries\rightmark}}
\rhead[\fancyplain{}{\bfseries\leftmark}]{\fancyplain{}{\bfseries\thepage}}
\cfoot{}
%
%
%
%
% Headers and footers for plain (blank) pages
\fancypagestyle{plain}{%
  \fancyhf{}%
  %\fancyhead[RO,LE]{\bfseries \thepage}%
  \fancyfoot[C]{\bfseries \thepage}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TOC / LOF / LOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\newcommand{\toc}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \setlength{\baselineskip}{2\baselineskip}
}
%
\newcommand{\toclof}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}
    \tableofcontents\cleardoublepage
    \listoffigures%
    \ifthenelse{ \equal{\loflot_in_toc}{0} }{\cleardoublepage}{%
%         \addcontentsline{toc}{chapter}{\numberline{}\listfigurename}\cleardoublepage%
        \addcontentsline{toc}{chapter}{\listfigurename}\cleardoublepage%
    }%
    \setlength{\baselineskip}{2\baselineskip}
}
%
% \addcontentsline must come before \cleardoublepage
\newcommand{\tocloflot}{
    \thispagestyle{empty}
    \setlength{\baselineskip}{0.5\baselineskip}%
    \tableofcontents\cleardoublepage%
    \listoffigures%
    \ifthenelse{ \equal{\loflot_in_toc}{0} }{\cleardoublepage}{%
%         \addcontentsline{toc}{chapter}{\numberline{}\listfigurename}\cleardoublepage%
        \addcontentsline{toc}{chapter}{\listfigurename}\cleardoublepage%
    }%
    \listoftables%
    \ifthenelse{ \equal{\loflot_in_toc}{0} }{\cleardoublepage}{%
%         \addcontentsline{toc}{chapter}{\numberline{}\listtablename}\cleardoublepage%
        \addcontentsline{toc}{chapter}{\listtablename}\cleardoublepage%
    }%
    \setlength{\baselineskip}{2\baselineskip}%
    \cleardoublepage
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Chapter with no numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\nonumchap}[1]{%
    \thispagestyle{empty}
    \phantomsection
%     \addcontentsline{toc}{chapter}{\numberline{}#1}
    \addcontentsline{toc}{chapter}{#1}
    \chapter*{#1}
    \chaptermark{#1}
    \markboth{#1}{#1}
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print nomenclature list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printnomencl}{%
\phantomsection
% \addcontentsline{toc}{chapter}{\numberline{}\nomname}
\addcontentsline{toc}{chapter}{\nomname}
\markboth{\nomname}{\nomname}
\printnomenclature[0.8in]
\cleardoublepage
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print bib list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printbib}[1]{%
\renewcommand{\bibname}{References}
% \setlength{\bibitemsep}{\baselineskip}
\cleardoublepage
\phantomsection
% \addcontentsline{toc}{chapter}{\numberline{}\bibname}
\addcontentsline{toc}{chapter}{\bibname}
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
%
\makeatletter
\newcommand\BackgroundPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vspace{5cm}
    %\vfill
    %\centering\includegraphics{#1}
%     \centering\includegraphics[height=\paperheight]{#1}
	\centering\includegraphics[width=0.8\paperwidth]{#1}
    \vfill
}}}
\newcommand\TuPicture[1]{%
    \setlength{\unitlength}{1pt}%
    \put(0,\strip@pt\paperheight){%
    \parbox[t][\paperheight]{\paperwidth}{%
    \vfill
%     \centering\includegraphics[width=\paperwidth]{#1}
    \includegraphics{#1}
}}}
\makeatother
%
\renewcommand{\maketitle}{%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main title with picture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{titlepage}
\ifthenelse{ \equal{\mscbackpicture}{} }{}{
\AddToShipoutPicture*{\BackgroundPicture{\mscbackpicture}}
}%
% \AddToShipoutPicture*{\TuPicture{./figs/TUD_bies_LR_A4_ENG}} % <- old logo
\ifpdf
  \AddToShipoutPicture*{\TuPicture{./figures/style/TU_border_A4_P_front_pdf}} % <- new logo, pdf format
\else
  \AddToShipoutPicture*{\TuPicture{./figures/style/TU_border_A4_P_front}} % <- new logo, eps format
\fi
    \begin{center}
    \large{\textsc{Master of Science Thesis}}%\\
    \end{center}%
    \newcommand{\HRule}{\rule{\linewidth}{1mm}}
    \setlength{\parindent}{0mm}
    \setlength{\parskip}{0mm}
    \vspace*{\stretch{4}}
    \HRule
    \begin{flushright}
    \huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[5mm]
%
    \Large{\textbf{\textsf{ \mscname}}}
    \end{flushright}
    \HRule
    \\ \mscdate \\
% 
    \vspace*{\stretch{1}}
 %
 % 
 \begin{center}
 \begin{tabular}{lcr}
 Faculty of \mscfaculty & $\cdot$ & Delft University of Technology\\
 \end{tabular}
 \end{center}
 \vspace{0.15cm}
\end{titlepage}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Second title page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\clearemptydoublepage
\thispagestyle{empty}
\setcounter{page}{1}
\begin{center}
\huge{\textbf{\textsf{\msctitle}}}\\ \Large{\textbf{\textsf{\mscsubtitle}}} \\[1cm]
\Large{\textsc{Master of Science Thesis}}\\[2cm]
\Large{For obtaining the degree of Master of Science in Aerospace Engineering at Delft University of Technology}\\[5cm]
\Large{\mscname}\\[1cm]
\Large{\mscdate}
\end{center}
\vspace*{\stretch{2}}
%
\begin{center}
\begin{tabular}{lcr}
Faculty of \mscfaculty & $\cdot$ & Delft University of Technology
\end{tabular}
\end{center}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\vspace*{15cm}
\vspace*{\stretch{3}}
\thispagestyle{empty}
%   \includegraphics{./figs/tud}\\  % <- old logo
\ifpdf
  \includegraphics{./figures/style/TU_d_line_P1_pdf}\\  % <- new logo, pdf format
\else
  \includegraphics{./figures/style/TU_d_line_P1}\\  % <- new logo, eps format
\fi
Copyright \copyright\ \mscname \\
% \newline\noindent
All rights reserved.
\cleardoublepage
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Readers Page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\thispagestyle{empty}
%
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
%
The undersigned hereby certify that they have read and recommend
to the Faculty of \mscfaculty\ for acceptance a thesis
entitled \textbf{``\msctitle''} by
\textbf{\mscname} in partial fulfillment of the
requirements for the degree of \textbf{Master of Science}.

\vspace{4cm}

\makebox[\textwidth][r]{Dated: \underline{ \mscdate } }
% \mbox{} \hfill Dated: \underline{ \mscdate }

\vspace{1cm}

\noindent

% vertical spacing distance between the readers
\newcommand{\vsp}{1.8cm}%
%
% head of department (first reader)
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreaderone}{} }{}{%
        Head of department:         \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreaderone}%
        \vfill%
    }
}
%
% daily supervisor (second reader)
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreadertwo}{} }{}{%
        Academic Supervisor:                 \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreadertwo}%
        \vfill%
    }
}
%
% guest, someone from outside the department/university (third reader)
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreaderthree}{} }{}{%
        Academic Supervisor:                 \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreaderthree}%
        \vfill%
    }
}
%
% person picked from a random room (fourth reader)
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreaderfour}{} }{}{%
        Industrial Supervisor:                  \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreaderfour}%
        \vfill%
    }
}
%
% in case you really want a big committee, get yourself a fifth reader
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreaderfive}{} }{}{%
        Reader:                     \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreaderfive}%
        \vfill%
    }
}
%
% and a sixth reader for a rediculously big reader
\parbox[t][\vsp][s]{\textwidth}{%
    \ifthenelse{ \equal{\mscreadersix}{} }{}{%
        Reader:                     \hfill \underline{ \hspace{7cm} }\\%
        \makebox[\textwidth][r]{\mscreadersix}%
        \vfill%
    }
}
%
\newpage\clearemptydoublepage
}
\makeindex
%\sloppy
\endinput
%%
%% End of file `dutmsc.cls'.
