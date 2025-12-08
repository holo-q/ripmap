; Classes
(class_declaration
  name: (identifier) @name.definition.class) @definition.class

(class
  name: (identifier) @name.definition.class) @definition.class

; Methods
(method_definition
  name: (property_identifier) @name.definition.method) @definition.method

; Functions
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

(generator_function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Arrow functions assigned to variables
(lexical_declaration
  (variable_declarator
    name: (identifier) @name.definition.function
    value: (arrow_function))) @definition.function

(variable_declaration
  (variable_declarator
    name: (identifier) @name.definition.function
    value: (arrow_function))) @definition.function

; Function expressions assigned to variables
(lexical_declaration
  (variable_declarator
    name: (identifier) @name.definition.function
    value: (function_expression))) @definition.function

(variable_declaration
  (variable_declarator
    name: (identifier) @name.definition.function
    value: (function_expression))) @definition.function

; Function calls
(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (member_expression
    property: (property_identifier) @name.reference.call)) @reference.call

; Constructor calls
(new_expression
  constructor: (identifier) @name.reference.class) @reference.class
