// modeled this build.sbt off of http://www.typesafe.com/activator/template/scala-library-seed#code/build.sbt

name := "pls"

version := "1.0.0"

scalaVersion := "2.11.7"

libraryDependencies  ++= Seq(
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test",

  "com.opencsv" % "opencsv" % "3.6",

  "org.scalanlp" %% "breeze" % "0.11.2",

  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.11.2"

  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  // "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

// bintray stuff
licenses += ("MIT", url("http://opensource.org/licenses/MIT"))
organization := "com.github.davidkellis"
