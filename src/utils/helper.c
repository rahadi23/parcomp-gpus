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

void parseArgsULong(char *arg, unsigned long *val)
{
  char *cp;
  unsigned long lVal;

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

  *val = (unsigned long)lVal;
}

void parseArgsFloat(char *arg, float *val)
{
  char *cp;
  float lVal;

  cp = arg;

  if (*cp == 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is an empty string\n", arg);
    exit(1);
  }

  lVal = strtof(cp, &cp);

  if (*cp != 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is not float -- '%s'\n", arg, cp);
    exit(1);
  }

  *val = (float)lVal;
}
