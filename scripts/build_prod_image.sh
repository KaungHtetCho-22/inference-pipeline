#!/bin/env bash

docker build \
	-t monsoon-audio-biodiversity:prod \
	--target prod \
	-f docker/Dockerfile .
