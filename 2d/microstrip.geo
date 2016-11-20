// Shielded microstrip line
coarse_lc = 4e-2;
fine_lc = 2e-2;

Point(1) = { 0, 0, 0, fine_lc };
Point(2) = { 1, 0, 0, coarse_lc };
Point(3) = { 1, 1, 0, coarse_lc };
Point(4) = { 0, 1, 0, coarse_lc };

Point(5) = { 0, 0.15, 0, fine_lc };
Point(6) = { 0.2, 0.15, 0, fine_lc };
Point(7) = { 0.2, 0.17, 0, fine_lc };
Point(8) = { 0, 0.17, 0, fine_lc };
Point(9) = { 1, 0.15, 0, coarse_lc };

Line(1) = {1, 2};
Line(2) = {2, 9};
Line(3) = {9, 6};
Line(4) = {6, 5};
Line(5) = {5, 1};

Line Loop(6) = {1, 2, 3, 4, 5};
Plane Surface(7) = {6};

Line(11) = {9, 3};
Line(12) = {3, 4};
Line(13) = {4, 8};
Line(14) = {8, 7};
Line(15) = {7, 6};
Line(16) = {6, 9};

Line loop(17) = {11, 12, 13, 14, 15, 16};
Plane Surface(18) = {17};

Physical Line("dirichlet-boundary") = {1, 2, 11, 12, 14, 15, 4};
Physical Line("third-kind-boundary") = {5, 13};
Physical Line("discontinuity") = {3, 16};
Physical Surface("whole-domain") = {7, 18};

