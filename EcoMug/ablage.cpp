    py::class_<EcoMug>(
        module_handle, "EcoMug"
        ).def(py::init<float>())
      .def_property("multiplier", &SomeClass::get_mult, &SomeClass::set_mult)
      .def("multiply", &SomeClass::multiply)
      .def("multiply_list", &SomeClass::multiply_list)
      // .def_property_readonly("image", &SomeClass::make_image)
      .def_property_readonly("image", [](SomeClass &self) {
                py::array out = py::cast(self.make_image());
                return out;
              })
      // .def("multiply_two", &SomeClass::multiply_two)
      .def("multiply_two", [](SomeClass &self, float one, float two) {
          return py::make_tuple(self.multiply(one), self.multiply(two));
        })
      .def("function_that_takes_a_while", &SomeClass::function_that_takes_a_while)
      ;
