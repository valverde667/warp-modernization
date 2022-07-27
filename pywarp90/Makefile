# Possible targets: (install is the default)
#     install, debug, clean, cleandebug: for serial
#     pinstall, pdebug, pclean, pcleandebug: for parallel
#     installall, debugall, cleanall: for serial and parallel

# For now, assume using python (instead of python3)
PYTHON = python3
-include Makefile.local

install:
	make -f Makefile.Forthon3 install PYTHON=$(PYTHON)

pinstall:
	make -f Makefile.Forthon3.pympi install PYTHON=$(PYTHON)

debug:
	make -f Makefile.Forthon3 install DEBUG='-g --farg "-O0"' BUILDBASEDIR=build3debug PYTHON=$(PYTHON)

pdebug:
	make -f Makefile.Forthon3.pympi install DEBUG='-g --farg "-O0"' BUILDBASEDIR=build3paralleldebug PYTHON=$(PYTHON)

clean:
	make -f Makefile.Forthon3 clean

pclean:
	make -f Makefile.Forthon3.pympi clean

cleandebug:
	make -f Makefile.Forthon3 clean BUILDBASEDIR=build3debug

pcleandebug:
	make -f Makefile.Forthon3.pympi clean BUILDBASEDIR=build3paralleldebug

installall: install pinstall

debugall: debug pdebug

cleanall: clean pclean cleandebug pcleandebug
	rm -rf build*

help:
	@echo "Possible targets: (install is the default)"
	@echo "    install, debug, clean, cleandebug: for serial"
	@echo "    pinstall, pdebug, pclean, pcleandebug: for parallel"
	@echo "    installall, debugall, cleanall: for serial and parallel"
	@echo "    help: this message"

