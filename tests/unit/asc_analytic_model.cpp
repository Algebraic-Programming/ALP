
#include "analytic_model.hpp"

#include <sstream>
#include <iostream>


int main() {
	{
		// this is a 1D problem over 10 cores and 1M problem size, with a fictional
		// cache size of 5000 bytes
		asc::AnalyticModel< 1, 1, false > am( 5000, {10}, {1000000}, {true} );
		// cannot test minor tensors for 1D problems (TODO test elsewhere)
		// add three float vectors
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		// suppose we just add them
		am.setNumStages( 1 );
		// this problem should be feasible:
		//  - every processing unit gets 100000 elements per vector
		//  - their byte size is 400000 per vector
		//  - there are three vectors of size 1200000 bytes total
		//  - block size that maximises reuse is 5000 / 12 = 416
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 1: suggested block size is " << bsize << ", ";
			if( bsize != 416 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 416, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 1: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 10;
		}
	}
	{
		// this is a 1D problem over a 2D 2 x 5 process mesh with otherwise the same
		// test parameters as the above test
		asc::AnalyticModel< 2, 1, false > am( 5000, {2,5}, {1000000}, {true} );
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		am.setNumStages( 1 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 2: suggested block size is " << bsize << ", ";
			if( bsize != 416 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 416, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 2: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 20;
		}
	}
	{
		// this is a 1D problem over a 5D 1 x 1 x 1 x 2 x 5 process mesh with
		// otherwise the same test parameters as the above test
		asc::AnalyticModel< 5, 1, false > am( 5000, {1,1,1,2,5}, {1000000}, {true} );
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		am.addGlobalTensor( 4, {true} );
		am.setNumStages( 1 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 3: suggested block size is " << bsize << ", ";
			if( bsize != 416 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 416, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 3: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 30;
		}
	}
	{
		// test a 1D case where a trivial solution is possible
		asc::AnalyticModel< 1, 1, false > am( 24000, {10}, {10000}, {true} );
		am.addGlobalTensor( 8, {true} );
		am.addGlobalTensor( 8, {true} );
		am.setNumStages( 1 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 4: suggested block size is " << bsize << ", ";
			if( bsize != 1000 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 10000, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 4: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 40;
		}
	}
	{
		// test a 1D case where a trivial solution is possible
		asc::AnalyticModel< 1, 1, false > am( 3003, {1}, {1001}, {true} );
		am.addGlobalTensor( 3, {true} );
		am.setNumStages( 1 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 5: suggested block size is " << bsize << ", ";
			if( bsize != 1001 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 1001, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 5: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 50;
		}
	}
	{
		// test for the other trivial (worst-case trivial) solution, 1D
		asc::AnalyticModel< 1, 1, false > am( 32, {8}, {2538791}, {true} );
		am.addGlobalTensor( 8, {true} );
		am.addGlobalTensor( 8, {true} );
		am.addGlobalTensor( 8, {true} );
		am.addGlobalTensor( 8, {true} );
		am.setNumStages( 2 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 6: suggested block size is " << bsize << ", ";
			if( bsize != 1 ) {
				std::cout << "x\n";
				std::ostringstream oss;
				oss << "Expected block size 1, got " << bsize << " instead";
				throw std::runtime_error( oss.str() );
			} else {
				std::cout << "v\n";
			}
		} catch( const std::exception &e ) {
			std::cerr << "Error during test case 6: " << e.what() << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 60;
		}
	}
	{
		// test with no feasible solution, 1D
		asc::AnalyticModel< 1, 1, false > am( 1, {8}, {2538791}, {true} );
		am.addGlobalTensor( 1, {true} );
		am.addGlobalTensor( 1, {true} );
		am.setNumStages( 1 );
		try {
			const size_t bsize = am.getBlockSize( 0 );
			std::cout << "Test case 7: suggested block size is " << bsize << ", x\n";
			std::cerr << "Error during test case 7: a blocksize was returned even "
				<< "though the problem is infeasible" << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 70;
		} catch( ... ) {
			std::cout << "Test case 7: infeasibility correctly detected\n";
		}
	}

	// done
	std::cout << "Test OK\n" << std::endl;
	return 0;
}

