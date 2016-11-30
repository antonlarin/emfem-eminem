// Discontinuity in parallel-plate waveguide
coarse_lc = 5e-3;
fine_lc = 2.5e-3;

Point(1) = { 0, 0, 0, coarse_lc };
Point(2) = { 0.1, 0, 0, fine_lc };
Point(3) = { 0.1, 0.0175, 0, fine_lc };
Point(4) = { 0.15, 0.0175, 0, fine_lc };
Point(5) = { 0.15, 0, 0, fine_lc };
Point(6) = { 0.25, 0, 0, coarse_lc };
Point(7) = { 0.25, 0.035, 0, coarse_lc };
Point(8) = { 0, 0.035, 0, coarse_lc };

// Lines for waveguide casing
Line(1) = { 1, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };
Line(4) = { 4, 5 };
Line(5) = { 5, 6 };
Line(6) = { 6, 7 };
Line(7) = { 7, 8 };
Line(8) = { 8, 1 };

Line Loop(9) = { 1, 2, 3, 4, 5, 6, 7, 8 };
Plane Surface(10) = { 9 };

// Dielectric rod
Line(11) = { 2, 5 };
Line(12) = { 5, 4 };
Line(13) = { 4, 3 };
Line(14) = { 3, 2 };

Line Loop(15) = { 11, 12, 13, 14 };
Plane Surface(16) = { 15 };

Physical Line("third-kind-boundary") = { 1, 11, 5, 6, 7, 8 };
Physical Line("discontinuity") = { 4, 3, 2, 12, 13, 14 };
Physical Surface("whole-domain") = { 10, 16 };

