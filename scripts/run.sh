#!/usr/bin/env bash

until $@; do echo retrying; done
