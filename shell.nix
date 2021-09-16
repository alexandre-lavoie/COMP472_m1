{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python38.withPackages (ps: with ps; [
    numpy
    matplotlib
    pandas
    scikit-learn
  ]);
in pkgs.mkShell {
  name = "COMP472";
  packages = [
    pythonEnv
  ];
}
