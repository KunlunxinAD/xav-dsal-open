
all: build

.PHONY: download
download:
	@if [ "$(USE_XPYTORCH_LOWER_VERSION)" = "TRUE" ]; then \
		echo "Pytorch version under 3.2, skipping download step."; \
	else \
		echo "! (Default) Using pytorch version over 3.2, download header files."; \
		if [ ! -d ./tmp ]; then \
			echo "Downloading dependency header files (hxpc)"; \
			mkdir -p ./tmp; \
			bcecmd bos cp bos:/klx-sdk-release-public/xav/xmlir/xhpc_header_file_0604.tar.gz ./tmp/xhpc.tar.gz; \
			mkdir -p tmp/include && tar -xzf ./tmp/xhpc.tar.gz -C ./tmp/include; \
		else \
			echo "Temp directory exists. Skipping download."; \
		fi; \
	fi

.PHONY: clean_download
clean_download:
	rm -rf ./tmp

.PHONY: build
build: | download
	cd python && python setup.py build

.PHONY: develop
develop: | download
	cd python && python setup.py develop

.PHONY: dist
dist: | download
	cd python && python setup.py bdist_wheel
	mkdir -p output && cp python/dist/* output/

.PHONY: clean
clean: clean_download
	- rm -rf c++/build
	-cd python && python setup.py clean
	-rm -rf python/build
	-rm -rf python/dist

