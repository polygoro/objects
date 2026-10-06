POLY ?= poly
SRC := geoframe.poly

SHAPES := triangle:3 square:4 pentagon:5 hexagon:6 octagon:8 decagon:10
STLS := $(foreach s,$(SHAPES),$(word 1,$(subst :, ,$(s))).stl)

.PHONY: all clean

all: $(STLS)

sides = $(word 2,$(subst :, ,$(filter $(1):%,$(SHAPES))))

%.stl: $(SRC)
	$(POLY) build $< -D sides=$(call sides,$*) -o $@

clean:
	rm -f $(STLS)
