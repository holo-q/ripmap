; Class definitions
(class_definition
  name: (identifier) @name.definition.class) @definition.class

; Function definitions with optional return type
(function_definition
  name: (identifier) @name.definition.function
  return_type: (type)? @return_type) @definition.function

; Method definitions (inside class)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @name.definition.method))) @definition.method

; Typed parameters - captures parameter name and type
(typed_parameter
  (identifier) @param.name
  type: (type) @param.type) @typed_param

; Simple function calls: foo()
(call
  function: (identifier) @name.reference.call) @reference.call

; Method calls: obj.method() - capture both receiver and method
(call
  function: (attribute
    object: (identifier) @receiver
    attribute: (identifier) @name.reference.method_call)) @reference.method_call

; Chained method calls: obj.foo().bar()
(call
  function: (attribute
    object: (call) @receiver_call
    attribute: (identifier) @name.reference.method_call)) @reference.method_call

; Import statements for cross-file resolution
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.name) @import

(import_statement
  name: (dotted_name) @import.name) @import
