using ParallelTestRunner
using TensorKitTensors

testsuite = ParallelTestRunner.find_tests(@__DIR__)
delete!(testsuite, "testsetup")

ParallelTestRunner.runtests(TensorKitTensors, ARGS; testsuite)
