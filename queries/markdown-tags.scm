; Markdown tags - extract headings as navigable symbols
; Headings become the structure of documentation
; Uses @name.definition.X pattern expected by the parser

; ATX headings (# style)
(atx_heading
  (atx_h1_marker)
  heading_content: (inline) @name.definition.section)

(atx_heading
  (atx_h2_marker)
  heading_content: (inline) @name.definition.section)

(atx_heading
  (atx_h3_marker)
  heading_content: (inline) @name.definition.subsection)

(atx_heading
  (atx_h4_marker)
  heading_content: (inline) @name.definition.subsection)

(atx_heading
  (atx_h5_marker)
  heading_content: (inline) @name.definition.subsection)

(atx_heading
  (atx_h6_marker)
  heading_content: (inline) @name.definition.subsection)

; Setext headings (underline style)
(setext_heading
  heading_content: (paragraph) @name.definition.section)

; Code blocks with language - useful for understanding what languages docs cover
(fenced_code_block
  (info_string) @name.reference.call)
