#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

void parseArgsInt(char *arg, int *val)
{
  char *cp;
  long lVal;

  cp = arg;

  if (*cp == 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is an empty string\n", arg);
    exit(1);
  }

  lVal = strtol(cp, &cp, 10);

  if (*cp != 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is not an integer -- '%s'\n", arg, cp);
    exit(1);
  }

  *val = (int)lVal;
}
