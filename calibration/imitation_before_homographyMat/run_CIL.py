import argparse
import logging

from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017, BasicExperimentSuite
from carla.driving_benchmark.experiment_suites import DAC2018

from agents.imitation.imitation_learning import ImitationLearning

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the scripts'
    )
    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )
    argparser.add_argument(
        '--dac-2018',
        action='store_true',
        help='If you want to benchmark the dac-2018 instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the given log name'
    )
    argparser.add_argument(
        '-f', '--folder-name',
        metavar='F',
        default='double-lines-width_go-straight',
        help='folder in which current experiment data would be stored'
    )
    argparser.add_argument(
        '-t', '--task',
        metavar='T',
        default='go-straight',
        help='task to perform: go-straight, turn-left, turn-right'
    )
    argparser.add_argument(
        '-i', '--iterations',
        metavar='I',
        default=349,
        type=int,
        help='total number of iterations'
    )
    argparser.add_argument(
        '-w', '--weather',
        metavar='W',
        default=1,
        type=int,
        help='Use weathers, 1- ClearNoon, 6 - HeavyRainNoon, 8 - ClearSunset'
    )
    argparser.add_argument(
    	'-s', '--sample',
    	type=int,
    	default=0
    )

    args = argparser.parse_args()
    #print(args.sample)
    #print(type(args.sample))
    #g=input()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    agent = ImitationLearning(args.city_name, args.avoid_stopping, args.folder_name, opt = args.sample)

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    experiment_suite = DAC2018(args.city_name, args.task, args.weather, args.iterations)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
