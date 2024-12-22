/*
  @license
	Rollup.js v4.27.0
	Fri, 15 Nov 2024 10:40:04 GMT - commit c035068dfebeb959a35a8acf3ff008a249e2af73

	https://github.com/rollup/rollup

	Released under the MIT License.
*/
export { version as VERSION, defineConfig, rollup, watch } from './shared/node-entry.js';
import './shared/parseAst.js';
import '../native.js';
import 'node:path';
import 'path';
import 'node:process';
import 'node:perf_hooks';
import 'node:fs/promises';
import 'tty';
