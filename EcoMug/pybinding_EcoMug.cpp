#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <../../EcoMug.h>
namespace py = pybind11;

PYBIND11_MODULE(EcoMug, m) {
  // module_handle.doc() = "I'm a docstring hehe";
  // module_handle.def("some_fn_python_name", &some_fn);

  py::class_<EcoMug>(m, "EcoMug_Class")
      .def(py::init<>())
      .def("SetUseSky", &EcoMug::SetUseSky)
      .def("SetUseCylinder", &EcoMug::SetUseCylinder)
      .def("SetUseHSphere", &EcoMug::SetUseHSphere)
      .def("SetSkySize", &EcoMug::SetSkySize)
      .def("SetSkyCenterPosition", &EcoMug::SetSkyCenterPosition)
      .def("Generate", &EcoMug::Generate)
      .def("GetGenerationPosition", &EcoMug::GetGenerationPosition)
      .def("GetGenerationMomentum", &EcoMug::GetGenerationMomentum)
      .def("GetGenerationTheta", &EcoMug::GetGenerationTheta)
      .def("GetGenerationPhi", &EcoMug::GetGenerationPhi)
      .def("GetCharge", &EcoMug::GetCharge)
      .def("SetSeed", &EcoMug::SetSeed)
      .def("SetMinimumMomentum", &EcoMug::SetMinimumMomentum)
      .def("SetMaximumMomentum", &EcoMug::SetMaximumMomentum)
      .def("SetMinimumTheta", &EcoMug::SetMinimumTheta)
      .def("SetMaximumTheta", &EcoMug::SetMaximumTheta)
      .def("SetMinimumPhi", &EcoMug::SetMinimumPhi)
      .def("SetMaximumPhi", &EcoMug::SetMaximumPhi)
    ;
}
// .def("GetGenerationMomentum", py::overload_cast<>(&EcoMug::GetGenerationMomentum))
      // .def("GetGenerationMomentum", py::overload_cast<std::array<double, 3>& momentum>(&EcoMug::GetGenerationMomentum))
      // .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
      // .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");
