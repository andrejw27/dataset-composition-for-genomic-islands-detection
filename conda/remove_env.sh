#!/usr/bin/env bash

read -p "Delete environment 'datacompos'? (y/n)" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
  conda remove --name datacompos --all
fi