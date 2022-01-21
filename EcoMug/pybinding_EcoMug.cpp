#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <../../EcoMug.h>
namespace py = pybind11;

PYBIND11_MODULE(EcoMug, m) {
  // module_handle.doc() = "I'm a docstring hehe";
  // module_handle.def("some_fn_python_name", &some_fn);

  py::class_<EcoMug>(m, "EcoMug")
      .def(py::init<>())

      .def("SetUseSky", &EcoMug::SetUseSky)
      .def("SetSkySize", &EcoMug::SetSkySize)
      .def("SetSkyCenterPosition", &EcoMug::SetSkyCenterPosition)
      
      .def("SetUseCylinder", &EcoMug::SetUseCylinder)
      .def("SetCylinderRadius", &EcoMug::SetCylinderRadius)
      .def("SetCylinderHeight", &EcoMug::SetCylinderHeight)
      .def("SetCylinderCenterPosition", &EcoMug::SetCylinderCenterPosition)
      .def("SetCylinderMinPositionPhi", &EcoMug::SetCylinderMinPositionPhi)
      .def("SetCylinderMaxPositionPhi", &EcoMug::SetCylinderMaxPositionPhi)
      .def("GetCylinderRadius", &EcoMug::GetCylinderRadius)
      .def("GetCylinderHeight", &EcoMug::GetCylinderHeight)
      .def("GetCylinderCenterPosition", &EcoMug::GetCylinderCenterPosition)

      .def("SetUseHSphere", &EcoMug::SetUseHSphere)
      .def("SetHSphereRadius", &EcoMug::SetHSphereRadius)
      .def("SetHSphereCenterPosition", &EcoMug::SetHSphereCenterPosition)
      .def("SetHSphereMinPositionPhi", &EcoMug::SetHSphereMinPositionPhi)
      .def("SetHSphereMaxPositionPhi", &EcoMug::SetHSphereMaxPositionPhi)
      .def("SetHSphereMinPositionTheta", &EcoMug::SetHSphereMinPositionTheta)
      .def("SetHSphereMaxPositionTheta", &EcoMug::SetHSphereMaxPositionTheta)
      .def("GetHSphereRadius", &EcoMug::GetHSphereRadius)
      .def("GetHSphereCenterPosition", &EcoMug::GetHSphereCenterPosition)


      .def("Generate", &EcoMug::Generate)
      .def("SetGenerationMethod", &EcoMug::SetGenerationMethod)
      .def("GetGenerationMethod", &EcoMug::GetGenerationMethod)
      .def("GenerateFromCustomJ", &EcoMug::GenerateFromCustomJ)
      .def("SetDifferentialFlux", &EcoMug::SetDifferentialFlux)
      .def("SetSeed", &EcoMug::SetSeed)

      .def("GetGenerationPosition", &EcoMug::GetGenerationPosition)
      .def("GetGenerationMomentum", &EcoMug::GetGenerationMomentum)
      .def("GetGenerationTheta", &EcoMug::GetGenerationTheta)
      .def("GetGenerationPhi", &EcoMug::GetGenerationPhi)
      .def("GetCharge", &EcoMug::GetCharge)
      
      .def("SetMinimumMomentum", &EcoMug::SetMinimumMomentum)
      .def("SetMaximumMomentum", &EcoMug::SetMaximumMomentum)
      .def("SetMinimumTheta", &EcoMug::SetMinimumTheta)
      .def("SetMaximumTheta", &EcoMug::SetMaximumTheta)
      .def("SetMinimumPhi", &EcoMug::SetMinimumPhi)
      .def("SetMaximumPhi", &EcoMug::SetMaximumPhi)

      .def("GetMinimumMomentum", &EcoMug::GetMinimumMomentum)
      .def("GetMaximumMomentum", &EcoMug::GetMaximumMomentum)
      .def("GetMinimumTheta", &EcoMug::GetMinimumTheta)
      .def("GetMaximumTheta", &EcoMug::GetMaximumTheta)
      .def("GetMinimumPhi", &EcoMug::GetMinimumPhi)
      .def("GetMaximumPhi", &EcoMug::GetMaximumPhi)
    ;
}
// .def("GetGenerationMomentum", py::overload_cast<>(&EcoMug::GetGenerationMomentum))
      // .def("GetGenerationMomentum", py::overload_cast<std::array<double, 3>& momentum>(&EcoMug::GetGenerationMomentum))
      // .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
      // .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");
