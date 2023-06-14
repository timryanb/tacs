# ============================================
#
# Make file for TACS_DIR/
#
# ============================================

include Makefile.in
include TACS_Common.mk

TACS_SUBDIRS = src \
	src/bpmat \
	src/elements \
	src/elements/dynamics \
	src/elements/basis \
	src/elements/shell \
	src/constitutive \
	src/functions \
	src/io

TACS_OBJS := $(addsuffix /*.o, ${TACS_SUBDIRS})

default:
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
	   echo "Building Complex TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) TACS_DIR=${TACS_DIR} TACS_DEF="${TACS_DEF} -DTACS_USE_COMPLEX") || exit 1; \
            done \
	else \
	   echo "Building Real TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) TACS_DIR=${TACS_DIR} TACS_DEF=${TACS_DEF}) || exit 1; \
            done \
	fi
	${CXX} ${SO_LINK_FLAGS} ${TACS_OBJS} ${TACS_EXTERN_LIBS} -o ${TACS_DIR}/lib/libtacs.${SO_EXT}
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
		echo "ctypedef complex TacsScalar" > tacs/TacsTypedefs.pxi; \
		echo "TACS_NPY_SCALAR = np.NPY_CDOUBLE" > tacs/TacsDefs.pxi; \
		echo "dtype = complex" >> tacs/TacsDefs.pxi; \
	else \
		echo "ctypedef double TacsScalar" > tacs/TacsTypedefs.pxi; \
		echo "TACS_NPY_SCALAR = np.NPY_DOUBLE" > tacs/TacsDefs.pxi; \
		echo "dtype = np.double" >> tacs/TacsDefs.pxi; \
	fi

debug:
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
	   echo "Building Complex TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) debug TACS_DIR=${TACS_DIR} TACS_DEF="${TACS_DEF} -DTACS_USE_COMPLEX") || exit 1; \
            done \
	else \
	   echo "Building Real TACS"; \
	   for subdir in $(TACS_SUBDIRS) ; do \
	      echo "making $@ in $$subdir"; \
	      echo; (cd $$subdir && $(MAKE) debug TACS_DIR=${TACS_DIR} TACS_DEF=${TACS_DEF}) || exit 1; \
            done \
	fi
	${CXX} ${SO_LINK_FLAGS} ${TACS_OBJS} ${TACS_EXTERN_LIBS} -o ${TACS_DIR}/lib/libtacs.${SO_EXT}
	@if [ "${TACS_IS_COMPLEX}" = "true" ]; then \
		echo "ctypedef complex TacsScalar" > tacs/TacsTypedefs.pxi; \
		echo "TACS_NPY_SCALAR = np.NPY_CDOUBLE" > tacs/TacsDefs.pxi; \
		echo "dtype = complex" >> tacs/TacsDefs.pxi; \
	else \
		echo "ctypedef double TacsScalar" > tacs/TacsTypedefs.pxi; \
		echo "TACS_NPY_SCALAR = np.NPY_DOUBLE" > tacs/TacsDefs.pxi; \
		echo "dtype = np.double" >> tacs/TacsDefs.pxi; \
	fi

complex: TACS_IS_COMPLEX=true
complex: default

complex_debug: TACS_IS_COMPLEX=true
complex_debug: debug

interface:
	CFLAGS="${PIP_CFLAGS}" ${PIP} install -e .\[all\];

complex_interface: PIP_CFLAGS+=-DTACS_USE_COMPLEX
complex_interface: interface

interface_debug: PIP_CFLAGS+=-DCYTHON_TRACE\=1
interface_debug: interface

complex_interface_debug: PIP_CFLAGS+=-DCYTHON_TRACE\=1
complex_interface_debug: complex_interface

clean:
	${RM} lib/libtacs.a lib/libtacs.${SO_EXT}
	${RM} tacs/*.${SO_EXT} tacs/*.cpp
	@for subdir in $(TACS_SUBDIRS) ; do \
	  echo "making $@ in $$subdir"; \
	  echo; \
	     (cd $$subdir && $(MAKE) $@ TACS_DIR=${TACS_DIR}) || exit 1; \
	done
