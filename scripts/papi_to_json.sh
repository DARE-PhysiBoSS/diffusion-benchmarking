#!/bin/bash

# Run papi_avail and process the output
papi_avail -a | awk '
BEGIN {
    print "{"
    first = 1
}
# Skip header lines and only process meaningful rows
$1 ~ /^PAPI_/ {
    name = $1
    code = $2
    deriv = $3

    code_int = strtonum(code)

    # Capture the rest of the line as description
    desc = ""
    for (i = 4; i <= NF; i++) {
        desc = desc $i " "
    }
    # Remove trailing space
    sub(/[ \t]+$/, "", desc)

    if (!first) {
        print ","
    }
    first = 0

    # Print JSON entry
    printf "  \"%s\": {\n", name
    printf "    \"Code\": %d,\n", code_int
    printf "    \"Deriv\": \"%s\",\n", deriv
    printf "    \"Description\": \"%s\"\n", desc
    printf "  }"
}
END {
    print "\n}"
}
'