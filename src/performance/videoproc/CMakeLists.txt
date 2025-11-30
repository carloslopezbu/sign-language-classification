cmake_minimum_required(VERSION 3.15)
project(videoproc_module)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Forzar uso del Python del entorno actual
set(Python_FIND_VIRTUALENV FIRST)
set(Python_FIND_STRATEGY LOCATION)

# Configurar C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# Encontrar Python (detectará automáticamente el venv activo)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Encontrar spdlog
find_package(spdlog CONFIG REQUIRED)

# Encontrar pybind11
# Si está instalado en el venv, CMake lo encontrará automáticamente
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED CONFIG HINTS ${pybind11_DIR})

# Encontrar OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Incluir directorio de headers del proyecto
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Definir archivos fuente
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Crear módulo Python con pybind11
pybind11_add_module(videoproc ${SOURCES})

# Enlazar OpenCV y spdlog
target_link_libraries(videoproc PRIVATE ${OpenCV_LIBS} spdlog::spdlog)

# Configurar RPATH para que encuentre las librerías compartidas en macOS
if(APPLE)
    set_target_properties(videoproc PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Obtener el directorio site-packages del venv
# Obtener el directorio site-packages CORRECTO usando sysconfig
# 'platlib' es específico para librerías compiladas/dependientes de plataforma
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Instalar el módulo en el venv
install(TARGETS videoproc
    LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
)


# Instalar el stub file (.pyi) para type hints
install(FILES videoproc.pyi
    DESTINATION ${PYTHON_SITE_PACKAGES}
)

# Mensaje informativo
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "pybind11 directory: ${pybind11_DIR}")
