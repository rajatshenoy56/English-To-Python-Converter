import FileInput from "./components/Input/FileInput";
import TextInput from "./components/Input/TextInput";
import AudioInput from "./components/Input/AudioInput";
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import CustomNavbar from "./components/Navbar/Navbar"
import FrontPage from "./components/FrontPage/FrontPage";
import { Fragment } from 'react';

const App = () => {
  return (
	<Router>
	<Fragment>
		<Switch>
		  <Route exact path={["/audioinput","/fileinput", "/textinput", "/"]}>
			<CustomNavbar/>
			  <Route exact path='/fileinput' component={FileInput} />
			  <Route exact path='/textinput' component={TextInput} />
			  <Route exact path='/audioinput' component={AudioInput} />
			  <Route exact path='/' component={FrontPage} />
		  </Route>
		</Switch>
	</Fragment>
	</Router>
  );
}

export default App;
