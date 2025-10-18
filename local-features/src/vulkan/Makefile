SRC_DIR := shaders
BUILD_DIR := shaders_build
COPY_DIR := $(BUILD_DIR)/source
EXPANDED_DIR := $(BUILD_DIR)

SRC_FILES := $(shell find $(SRC_DIR) -type f -path '*.comp')
HEADER_FILES = $(shell find $(SRC_DIR) -type f -path '*.glsl')

TARGETS := $(patsubst $(SRC_DIR)/%,$(EXPANDED_DIR)/%,$(SRC_FILES))
HEADER_TARGETS := $(patsubst $(SRC_DIR)/%,$(EXPANDED_DIR)/%,$(HEADER_FILES))

SPV_TARGETS := $(patsubst $(SRC_DIR)/%.comp,$(BUILD_DIR)/spv/%.spv,$(SRC_FILES))

all: $(TARGETS) $(HEADER_TARGETS) $(SPV_TARGETS)

GLSLC_FLAGS := -DPRECISION_FLOAT32
GLSLC := glslc $(GLSLC_FLAGS)
GPP := gpp -I$(COPY_DIR) -U "@" "" "(" ";" ")" "(" ")" "\#" "\\" -M "\$$" "" "\W[\W" "\W;\W" "\W]" "[" "]" 

DEP_FILES := $(TARGETS:.comp=.comp.d)

$(COPY_DIR)/%: $(SRC_DIR)/%
	@mkdir -p $(dir $@)
	cp $< $@

$(EXPANDED_DIR)/%.gpp.glsl: $(COPY_DIR)/%.gpp.glsl $(COPY_DIR)/macros.gpp
	@mkdir -p $(dir $@)
	@# awk to remove leading blank lines from #include macros
	set -euo pipefail; $(GPP) $< | bash trim_blank.sh > $@

$(EXPANDED_DIR)/%.gpp.comp: $(COPY_DIR)/%.gpp.comp $(COPY_DIR)/macros.gpp
	@mkdir -p $(dir $@)
	@ # awk to remove leading blank lines from #include macros, so error line numbers are not offset
	set -euo pipefail; $(GPP) $< | bash trim_blank.sh > $@

$(EXPANDED_DIR)/%: $(COPY_DIR)/%
	@mkdir -p $(dir $@)
	cp $< $@

$(EXPANDED_DIR)/%.comp.d : $(EXPANDED_DIR)/%.comp $(HEADER_TARGETS)
	@mkdir -p $(dir $@)
	$(GLSLC) -MM -MT $(patsubst $(EXPANDED_DIR)/%.comp.d,$(BUILD_DIR)/spv/%.spv,$@) $< > $@

-include $(DEP_FILES)

$(BUILD_DIR)/spv/%.spv : $(EXPANDED_DIR)/%.comp
	@mkdir -p $(dir $@)
	$(GLSLC) -c -fshader-stage=comp -O --target-env=vulkan1.2 $< -o $@

.PRECIOUS: $(EXPANDED_DIR)/%.comp.d
